import math
import itertools
import re

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import gc
import random
import numpy as np
import pandas as pd
import sympy
from sympy import Eq

from scipy.special import comb
import scipy.optimize as opt

from symengine import sympify as se_sympify
import yaml

from utils.stages import load_yaml_config, run_stages
from utils.data import expr_to_Y_pred
from utils.evaluate import get_sympy_complexity
from utils.exprutils import time_limit, TimeoutException, has_nested_func

from .token_generator.mcts import MCTS_TokenGenerator
from .token_generator.gp import GP_TokenGenerator
from .token_generator.random import Random_TokenGenerator

from .operators import (
    Identity_op,
    Sin_op,
    Cos_op,
    Exp_op,
    Log_op,
    Neg_op,
    Inv_op,
    Add_op,
    Mul_op,
    Sub_op,
    Div_op,
    SemiDiv_op,
    SemiSub_op,
)

from .operators import (
    Sign_op,
    Pow2_op,
    Pow3_op,
    Pow_op,
    Sigmoid_op,
    Abs_op,
    Cosh_op,
    Tanh_op,
    Sqrt_op,
)

from .models import PSRN

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def insert_B_on_Add(expr_sympy):
    """add bias terms to Add node

    Examples
    ========

    >>> expr1 + expr2 -> expr1 + expr2 + B

    """
    cnt_B = 0

    def do(x):
        B = sympy.Symbol("B")
        nonlocal cnt_B
        cnt_B += 1
        return x.func(*(x.args + (B,)))

    expr_sympy = expr_sympy.replace(lambda x: x.is_Add, lambda x: do(x))
    return expr_sympy


def condense(eq, *x):
    """collapse additive/multiplicative constants into single
    variables, returning condensed expression and replacement
    values.

    https://stackoverflow.com/questions/71315789/
    """
    reps = {}
    con = sympy.numbered_symbols("c")
    free = eq.free_symbols

    def c():
        while True:
            rv = next(con)
            if rv not in free:
                return rv

    def do(e):
        i, d = e.as_independent(*x)
        if not i.args:
            return e
        return e.func(reps.get(i, reps.setdefault(i, c())), d)

    rv = eq.replace(lambda x: x.is_Add or x.is_Mul, lambda x: do(x))
    reps = {v: k for k, v in reps.items()}
    keep = rv.free_symbols & set(reps)
    reps = {k: reps[k].xreplace(reps) for k in keep}
    return rv, reps


def densify(expr_c, variables):
    variables_sympy = [sympy.Symbol(v) for v in variables]
    expr_c_sympy = sympy.sympify(expr_c)
    expr_c_dense_sympy, dense_dict = condense(expr_c_sympy, *variables_sympy)
    value_ls = []
    name_ls = []
    for key, value in zip(dense_dict.keys(), dense_dict.values()):
        for f in value.free_symbols:
            value = value.subs(f, 1.0)
        value_ls.append(value)
        name_ls.append(key)
    for atom in expr_c_dense_sympy.atoms():
        if "C" in str(atom):
            value_ls.append(1.0)
            name_ls.append(atom)
    dict_final = {}
    for i in range(len(value_ls)):
        dict_final[name_ls[i]] = value_ls[i]
    return expr_c_dense_sympy, dict_final


def finallize_const_name(expr_dense_sympy, dict_final, add_bias=True):
    cnt_c = 0
    dict_final_final = {}
    for atom in expr_dense_sympy.atoms():
        if "C" in str(atom) or "c" in str(atom) or "B" in str(atom):
            new_atom = sympy.Symbol("a{}".format(cnt_c))
            expr_dense_sympy = expr_dense_sympy.subs(atom, new_atom)
            cnt_c += 1
            dict_final_final[new_atom] = dict_final[atom]
    if add_bias:
        if expr_dense_sympy.func is not sympy.core.add.Add:
            new_atom = sympy.Symbol("a{}".format(cnt_c))
            expr_dense_sympy += se_sympify(new_atom)
            dict_final_final[new_atom] = 0.0
            cnt_c += 1
    return expr_dense_sympy, dict_final_final


def replace_c_with_a(expr_dense_sympy):
    # a0 + a1*qdot + a2*tau + a4*qdot*log(Abs(a3*qdot))
    """
    replace all a constant to C constant in expressioin

    Examples
    ========
    Input:
    >>> a0 * sin(x) + a1
    Output:
    >>> C0 * sin(x) + C1

    """
    for atom in expr_dense_sympy.atoms():
        str_atom = str(atom)
        if "a" == str_atom[0] and len(str_atom) >= 2 and str_atom[1:].isdigit():
            new_atom = sympy.Symbol("C{}".format(str_atom[1:]))
            expr_dense_sympy = expr_dense_sympy.subs(atom, new_atom)
    return expr_dense_sympy


def is_const(expr_sympy):
    """judge whether sub expression is const

    Examples
    ========
    Input:
    >>> exp(2)
    Output:
    >>> True

    Input:
    >>> exp(x)
    Output:
    >>> False

    """
    val = expr_sympy.n(1)
    if isinstance(val, sympy.core.numbers.Float) and (not expr_sympy.is_Number):
        return True
    else:
        return False


def replace_evaluatable(expr):
    """
    Find the result of all the evaluatable parts of the expression
    """
    replace_map = {}
    for subexpr in expr.find(is_const):
        val = subexpr.evalf()
        replace_map[subexpr] = val
    return expr.subs(replace_map, simultaneous=True)


def replace_exponent(expression):
    pattern = r"\*\*(\((.*?)\)|(\d+(\.\d+)?))"

    def replace_func(match):
        exp = match.group(2) or match.group(3)
        if exp.strip().replace(".", "").isdigit():
            return f"**(C*{exp})"
        else:
            return f"**(C*({exp}))"

    replaced_expression = re.sub(pattern, replace_func, expression)

    pattern = r"\*\*(\d+(\.\d+)?)(?!\d)"

    def replace_func(match):
        exp = match.group(1)
        if exp.strip().replace(".", "").isdigit():
            return f"**(C*{exp})"
        else:
            return f"**(C*({exp}))"

    replaced_expression = re.sub(pattern, replace_func, replaced_expression)

    return replaced_expression


def to_C_expr(expr, variables, use_replace_exponent=False):
    """
    1. Add a constant C to the front of all operators except exp and Abs
    2. Replace all variables with c* variables
    """
    expr_num = replace_evaluatable(expr)
    expr_num = str(expr_num)

    if use_replace_exponent:
        expr_num = replace_exponent(expr_num)
        print("replaced:", expr_num)

    ops = ["sin", "cos", "tan", "log", "asin", "acos", "atan", "sign"]
    for op in ops:
        expr_num = expr_num.replace(op, "C*{}".format(op))
    for variable in variables:
        expr_num = re.sub(
            r"(?<![a-zA-Z]){}(?![a-zA-Z])".format(variable),
            r"(C*{})".format(variable),
            expr_num,
        )

    cnt_C = 0

    def replace_C(matched):
        nonlocal cnt_C
        cnt_C += 1
        return "C{}".format(cnt_C - 1)

    expr_num = re.sub(r"C", replace_C, expr_num)
    return expr_num


def replace_B(expr_c_sympy):
    """
    Replace all the B constants with the Bi form
    """
    cnt_B = 0

    def replace_C(matched):
        nonlocal cnt_B
        cnt_B += 1
        return "B{}".format(cnt_B - 1)

    expr_c_sympy = re.sub(r"B", replace_C, expr_c_sympy)
    return expr_c_sympy, cnt_B


def get_expr_C_and_C0(expr, variables, add_bias=True, use_replace_exponent=False):
    """Converts the expr_num string expression
    into an expr_c expression with a constant placeholder
    and the corresponding constant initial value C0
    """
    expr_sympy = sympy.sympify(expr)
    expr_c = to_C_expr(expr_sympy, variables, use_replace_exponent=use_replace_exponent)
    expr_c_sympy = sympy.sympify(expr_c)
    expr_c_sympy, dict_c = densify(expr_c_sympy, variables)
    if add_bias:
        expr_c_sympy = insert_B_on_Add(expr_c_sympy)
    expr_c_sympy_str = str(expr_c_sympy)
    expr_c_sympy_str, cnt_B = replace_B(expr_c_sympy_str)
    for i in range(cnt_B):
        dict_c[sympy.Symbol("B{}".format(i))] = 0.0
    expr_c_sympy = sympy.sympify(expr_c_sympy_str)
    expr_dense_sympy, dict_final = finallize_const_name(
        expr_c_sympy, dict_c, add_bias=True
    )
    expr_final_sympy = replace_c_with_a(expr_dense_sympy)
    C0 = np.array(list(dict_final.values()))
    # return str(expr_final_sympy), C0
    return expr_final_sympy, C0  #


def set_real(expr_c_sympy):
    """Set all free variables of the expression to real numbers"""
    expr_c_sympy = sympy.sympify(expr_c_sympy)
    for var in expr_c_sympy.free_symbols:
        expr_c_sympy = expr_c_sympy.subs(var, sympy.Symbol(str(var), real=True))
    return expr_c_sympy


def remove_one_coeffs(expr):
    def traverse(arg):
        if arg.is_Atom:
            if arg == 1.0:
                return sympy.nsimplify(arg)
            else:
                return arg
        else:
            return arg.func(*[traverse(a) for a in arg.args])

    return traverse(expr)


def prun_constant(expr_num_sympy, n_digits=6):
    """The constants of an expression are
    rounded by precision and take
    the value 0 for numbers less than 1e-n in absolute value"""
    epsilon = 10.0 ** (-n_digits)

    def process_term(term):
        if isinstance(term, sympy.Float) or isinstance(term, sympy.Rational):
            float_value = float(term)
            if abs(float_value) < epsilon:
                return sympy.sympify("0")
            else:
                return sympy.Float(round(float_value, n_digits))
        elif isinstance(term, sympy.Mul):
            coeff, rest = term.as_coeff_Mul()
            if isinstance(coeff, (sympy.Float, sympy.Rational)):
                float_value = float(coeff)
                if abs(float_value) < epsilon:
                    return sympy.sympify("0")
                else:
                    rounded_value = sympy.Float(round(float_value, n_digits))
                    return rounded_value * rest
        return term

    return expr_num_sympy.replace(
        lambda x: isinstance(x, (sympy.Float, sympy.Rational, sympy.Mul)), process_term
    )


def recal_MSE(expr_str, X, Y, variables):
    """Recalculate the MSE of the expression with numpy"""
    functions = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sign": np.sign,
        "e": np.exp(1),
        "pi": np.pi,
    }
    try:
        values = {variables[j]: X[:, j] for j in range(X.shape[1])}
        pred = eval(expr_str.lower(), functions, values)
        true = Y[:, 0]
        diff = true - pred
        square = diff**2
        return np.mean(square)
    except Exception as e:
        print("recalMSE Error: ", e)
        print("expr_str      : ", expr_str)
        return np.nan


def has_irregular_power(s):
    pattern = r"\*\*\d+\.\d*[1-9]\d+[1-9]"
    return bool(re.search(pattern, s))


def has_large_integer(expr):
    if isinstance(expr, str):
        expr = sympy.S(expr)
    for atom in expr.atoms():
        if isinstance(atom, sympy.Integer) and abs(int(atom)) > 10:
            return True
        if isinstance(atom, sympy.Rational) and (abs(atom.p) > 10 or abs(atom.q) > 10):
            return True

    return False


from sympy import Add, Mul, Pow, Function


def crossover_expressions(expr1, expr2):
    if isinstance(expr1, (Add, Mul)) and isinstance(expr2, (Add, Mul)):
        args1 = list(expr1.args)
        args2 = list(expr2.args)
        idx1 = random.randint(0, len(args1) - 1)
        idx2 = random.randint(0, len(args2) - 1)
        args1[idx1], args2[idx2] = args2[idx2], args1[idx1]
        offspring1 = expr1.func(*args1)
        offspring2 = expr2.func(*args2)
        return offspring1, offspring2
    elif isinstance(expr1, Function) and isinstance(expr2, Function):
        offspring1 = expr1.func(expr2.args[0])
        offspring2 = expr2.func(expr1.args[0])
        return offspring1, offspring2
    elif isinstance(expr1, Pow) and isinstance(expr2, Pow):
        offspring1 = expr1.func(expr2.args[0])
        offspring2 = expr2.func(expr1.args[0])
        return offspring1, offspring2
    else:
        return expr1, expr2


def generate_crossover_expressions(expressions, size):
    ret = []
    while len(ret) < size:
        try:
            idx1, idx2 = random.sample(range(len(expressions)), 2)
        except ValueError:
            return ret
        expr1 = expressions[idx1]
        expr2 = expressions[idx2]
        offspring1, offspring2 = crossover_expressions(expr1, expr2)
        ret.extend([offspring1, offspring2])
    return ret[:size]


class PSRN_Regressor(nn.Module):
    """
    PSRN Regressor.

    Examples
    ========
    >>> #   input data
    >>> #  ____^______
    >>> # /           |
    >>> # ['x',      'y',   'x+x',        '1.5',      '3']
    >>> # <-n_variables->
    >>> #                 <-n_cross->
    >>> #                            <-trying_const_num->
    >>> # <-----------------n_inputs--------------------->

    """

    def __init__(
        self,
        variables=["x"],
        operators=None,
        n_symbol_layers=3,
        n_inputs=None,
        use_dr_mask=True,
        dr_mask_dir="./dr_mask",
        use_const=False,
        use_extra_const=False,
        n_sample_variables=None,
        stage_config="model/stages_config/benchmark.yaml",  # model/stages_config/chaotic.yaml
        token_generator_config="token_generator_config.yaml",
        token_generator="GP",  # MCTS / GP / Random / ...
        optimizer="Nelder-Mead",
        device="cuda",
    ):
        super(PSRN_Regressor, self).__init__()

        # self.stage_config = 'model/stages_config/benchmark.yaml'
        if isinstance(stage_config, str):
            self.stages_config = load_yaml_config(stage_config)
        elif isinstance(stage_config, dict):
            self.stages_config = stage_config

        if isinstance(token_generator_config, str):
            self.config = load_yaml_config(token_generator_config)
        elif isinstance(token_generator_config, dict):
            self.config = token_generator_config

        assert token_generator in ["GP", "MCTS", "Random"]

        if n_sample_variables is None:
            self.n_sample_variables = self.stages_config["default"]["n_psrn_inputs"]
        else:
            self.n_sample_variables = n_sample_variables
        self.token_generator_name = token_generator

        print("self.config")
        print(self.config)

        self.triu_ls = []
        if n_inputs is None:
            self.n_inputs = self.stages_config["default"]["n_psrn_inputs"]
        else:
            self.n_inputs = n_inputs
        for i in range(10):
            self.triu_ls.append(
                torch.triu_indices(
                    i + 1, i + 1, offset=0, dtype=torch.long, device=device
                )
            )

        self.N = 1
        self.use_const = use_const
        self.variables_repr = variables
        if operators is None:
            self.operators = self.stages_config["default"]["operators"]
        else:
            self.operators = operators
        self.n_symbol_layers = n_symbol_layers
        self.n_variables = len(variables)

        forbidden_pattern = r"[cCB]\d*"
        assert not any(
            [re.match(forbidden_pattern, var) for var in variables]
        ), "you cannot use c, C, or B as variables in regressor"

        self.operators_op = []
        for op_str in self.operators:
            op = eval(op_str + "_op")
            self.operators_op.append(op())

        self.use_dr_mask = use_dr_mask

        if self.use_dr_mask:
            self.dr_mask_dir = dr_mask_dir
            if not os.path.exists(self.dr_mask_dir):
                raise ValueError(
                    "dr_mask_dir not exist, got {}".format(self.dr_mask_dir)
                )
            file_name_mask = f'{self.n_symbol_layers}_{self.n_inputs}_[{"_".join(self.operators)}]_mask.npy'
            self.dr_mask_path = self.dr_mask_dir + "/" + file_name_mask
            if not os.path.exists(self.dr_mask_path):
                cmd = 'python utils/gen_dr_mask.py --n_symbol_layers={} --n_inputs={} --ops="{}"'.format(
                    self.n_symbol_layers,
                    self.n_inputs,
                    str(self.operators).replace(" ", ""),
                )
                print(
                    "dr_mask file not exist, got {}.\nPlease run `{}`".format(
                        self.dr_mask_path, cmd
                    )
                )
                print("=" * 40)
                print("Executing Automatically (dr mask gen) ....")
                print("cmd:")
                print(cmd)
                os.system(cmd)
                print("Execute finished (dr mask gen) ....")
                print("=" * 40)

            print("loading drmask from ", self.dr_mask_path)
            dr_mask = np.load(self.dr_mask_path)
            print("load finished")
            dr_mask = torch.from_numpy(dr_mask)
            print("to numpy finished")
            assert dr_mask.dim() == 1, "dr_mask should be 1-dim, got {}".format(
                dr_mask.dim()
            )
        else:
            dr_mask = None
            print("[INFO] use_dr_mask=False. May use more VRAM.")

        print(self.stages_config)

        self.net = PSRN(
            n_variables=self.n_inputs,
            operators=self.operators,
            n_symbol_layers=n_symbol_layers,
            dr_mask=dr_mask,
            device=device,
        )

        self.device = self.net.device
        self.operators_u = []
        self.operators_b = []
        for func in self.net.list[-1].list:
            if func.is_unary:
                self.operators_u.append(func._get_name())
            else:
                self.operators_b.append(func._get_name())
        self.hash_set = set()
        self.value_dict = {}
        self.X = None
        self.Y = None
        self.threshold = None
        self.real_time_display = None
        self.real_time_display_freq = None
        self.real_time_display_ntop = None
        self.prun_const = None
        self.prun_ndigit = None
        self.n_step_simulation = None
        self.probe = None
        self.probe_evalf = None
        self.n_down_sample = None
        self.pareto_frontier = None
        self.use_replace_expo = None
        self.use_strict_pareto = True
        self.use_extra_const = use_extra_const
        self.trying_const_range = [-3, 3]
        self.optimizer = optimizer

    def load_dr_mask(self):

        if self.use_dr_mask:
            if not os.path.exists(self.dr_mask_dir):
                raise ValueError(
                    "dr_mask_dir not exist, got {}".format(self.dr_mask_dir)
                )
            file_name_mask = f'{self.n_symbol_layers}_{self.n_inputs}_[{"_".join(self.operators)}]_mask.npy'
            self.dr_mask_path = self.dr_mask_dir + "/" + file_name_mask
            if not os.path.exists(self.dr_mask_path):
                cmd = 'python utils/gen_dr_mask.py --n_symbol_layers={} --n_inputs={} --ops="{}"'.format(
                    self.n_symbol_layers,
                    self.n_inputs,
                    str(self.operators).replace(" ", ""),
                )
                print(
                    "dr_mask file not exist, got {}.\nPlease run `{}`".format(
                        self.dr_mask_path, cmd
                    )
                )
                print("=" * 40)
                print("Executing Automatically (dr mask gen) ....")
                print("cmd:")
                print(cmd)
                os.system(cmd)
                print("Execute finished (dr mask gen) ....")
                print("=" * 40)

            print("loading drmask from ", self.dr_mask_path)
            dr_mask = np.load(self.dr_mask_path)
            print("load finished")
            dr_mask = torch.from_numpy(dr_mask)
            print("to numpy finished")
            assert dr_mask.dim() == 1, "dr_mask should be 1-dim, got {}".format(
                dr_mask.dim()
            )
        else:
            dr_mask = None
            print("[INFO] use_dr_mask=False. May use more VRAM.")

        return dr_mask

    def fit(
        self,
        X,
        Y,
        n_down_sample=20,
        eta=0.99,
        use_threshold=True,
        threshold=1e-10,
        probe=None,
        prun_const=True,
        prun_ndigit=6,
        real_time_display=True,
        real_time_display_freq=1,
        real_time_display_ntop=20,
        add_bias=True,
        together=False,
        top_k=30,
        use_replace_expo=False,
        use_strict_pareto=True,
        use_extra_const=False,
    ):
        """fitting data `X (n,m)` and `Y (n,1)` that
        >>> Y = F(X)

        Example
        =======
        >>> flag, pareto_frontier = regressor.fit(X,Y)
        """

        self.triu_ls = []
        for i in range(10):
            self.triu_ls.append(
                torch.triu_indices(
                    i + 1, i + 1, offset=0, dtype=torch.long, device=self.device
                )
            )
        # self.drm
        print("len(self.triu_ls):")
        print(len(self.triu_ls))
        print("=" * 40)

        print("num of samples:", len(X))
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["x{}".format(i) for i in range(X.shape[1])])
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

            X = X.values
            X = torch.from_numpy(X).float()

        else:
            self.feature_names = ["x{}".format(i) for i in range(X.shape[1])]

        self.n_variables = len(self.feature_names)
        self.variables = ["x{}".format(i) for i in range(self.n_variables)]
        self.variables_to_variables_repr_dict = {
            k: v for k, v in zip(self.variables, self.variables_repr)
        }
        self.variables_repr_to_variables_dict = {
            v: k for k, v in zip(self.variables, self.variables_repr)
        }

        print("variables: ", self.variables)
        print("variables repr: ", self.variables_repr)
        print(
            "self.variables_to_variables_repr_dict",
            self.variables_to_variables_repr_dict,
        )

        if isinstance(Y, pd.DataFrame):
            Y = Y.values
            Y = torch.from_numpy(Y).float().reshape(-1, 1)
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float().reshape(-1, 1)
        assert isinstance(
            X, torch.Tensor
        ), "X must be torch tensor, got {}, X:\n\n {}".format(type(X), X)
        assert isinstance(Y, torch.Tensor), "Y must be torch tensor, got {}".format(
            type(Y)
        )

        assert (
            X.shape[0] == Y.shape[0]
        ), "X.shape[0] must be equal to Y.shape[0], got {} and {}".format(
            X.shape[0], Y.shape[0]
        )
        assert Y.shape[1] == 1, "Y.shape[0] must be equal to 1, got {}".format(
            Y.shape[1]
        )

        X = X.to(self.device)
        Y = Y.to(self.device)

        self.X = X
        self.Y = Y

        print("=" * 40)

        self.n_down_sample = n_down_sample

        if n_down_sample is None or n_down_sample <= 0 or n_down_sample >= len(X):

            print("[INFO]: Down sampling disabled.")
            print(
                "[INFO]: PSRN forwarding will use {}/{} samples.".format(
                    n_down_sample, len(self.X)
                )
            )
        else:

            assert n_down_sample <= len(
                self.X
            ), "n_down_sample should be less than len(X), got {} and {}".format(
                n_down_sample, len(self.X)
            )

            print("[INFO]: Using down sampling,")

            print(
                "[INFO]: PSRN forwarding will use {}/{} samples to speed up".format(
                    n_down_sample, len(self.X)
                )
            )
            if self.use_const:
                print(
                    "[INFO]: Least Square will use all {} samples".format(len(self.X))
                )

        self.use_threshold = use_threshold
        if use_threshold:
            print(
                "[INFO]: Using threshold. Algo will stop when MSE < threshold: {}".format(
                    self.threshold
                )
            )
        else:
            print("[INFO]: Not using threshold.")
        self.threshold = threshold
        self.real_time_display = real_time_display
        self.real_time_display_freq = real_time_display_freq
        self.real_time_display_ntop = real_time_display_ntop
        self.prun_const = prun_const if self.use_const else False
        self.prun_ndigit = prun_ndigit
        self.eta = eta
        self.pareto_frontier = []

        self.add_bias = add_bias
        self.together = together

        self.top_k = top_k
        self.fitted_expr_c_set = set()

        self.use_replace_expo = use_replace_expo

        self.use_strict_pareto = use_strict_pareto

        if X.min() >= 0:
            self.is_positive = True
            print("[INFO]: Input is all positive.")
        else:
            self.is_positive = False

        if probe is not None:
            self.probe = self.my_simplify(
                self.replace_variables(probe, to_repr=False), self.together
            )
            self.probe_evalf = (
                self.set_real(self.remove_one_coeffs(self.probe), self.is_positive)
            ).evalf()
        else:
            self.probe = None

        if real_time_display:
            print("=" * 60)
            print(
                "[INFO]: Will display best {} expression per {} iterations".format(
                    real_time_display_ntop, real_time_display_freq
                )
            )

        if probe is not None:
            print("=" * 60)
            print(
                "[INFO]: Using benchmarking mode. Algo will stop when find expression (probe):"
            )
            print("Input (repr) --> ", probe)
            print("Sympy        --> ", self.probe)
            print("=" * 60)

        input_expr_ls = self.variables


        flag, pareto = run_stages(self.stages_config, self)
        return flag, pareto

    def fit_one(self, operators, n_psrn_inputs, n_sample_variables, time_limit):

        self.start_time = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        self.time_limit = time_limit

        self.operators = operators
        self.n_inputs = n_psrn_inputs
        self.n_sample_variables = n_sample_variables

        self.operators_op = []
        for op_str in self.operators:
            op = eval(op_str + "_op")
            self.operators_op.append(op())

        print("self.n_variables", self.n_variables)
        print("operators", self.operators)
        print("n_symbol_layers", self.n_symbol_layers)

        print("start building PSRN ...")
        print("self.n_variables", self.n_variables)
        print("self.operators", self.operators)
        print("self.n_inputs", self.n_inputs)

        if self.token_generator_name == "MCTS":
            self.n_sample_variables = 0
        else:
            self.n_sample_variables = min(self.n_sample_variables, self.n_variables)

        self.token_generator = eval(self.token_generator_name + "_TokenGenerator")(
            regressor=self,
            config=self.config[self.token_generator_name],
            variables=self.variables,
            operators_op=self.operators_op,
            use_const=self.use_const,
            n_inputs=self.n_inputs,
            use_extra_const=self.use_extra_const,
        )

        print("token generator:", self.token_generator)

        dr_mask = self.load_dr_mask()

        if (
            self.net.n_variables == self.n_inputs
            and self.net.operators == self.operators
            and self.net.n_symbol_layers == self.n_symbol_layers
        ):
            print("PSRN is the same, skip init")
        else:
            print(self.net.n_variables, self.n_inputs)
            print(self.net.operators, self.operators)
            print(self.net.n_symbol_layers, self.n_symbol_layers)

            self.net = PSRN(
                n_variables=self.n_inputs,
                operators=self.operators,
                n_symbol_layers=self.n_symbol_layers,
                dr_mask=dr_mask,
                device=self.device,
            )

        print("finished building PSRN")
        self.operators_u = []
        self.operators_b = []
        for func_name in self.operators:
            func_name_op = func_name + "_op"
            func = eval(func_name_op)()
            if func.is_unary:
                self.operators_u.append(func_name)
            else:
                self.operators_b.append(func_name)

        if self.use_const:
            # Constant and linear fitting is performed first
            fitted_c = np.mean(self.Y.cpu().numpy())
            fitted_c_expr = str(fitted_c)
            fitted_c_mse = np.mean((self.Y.cpu().numpy() - fitted_c) ** 2)
            complexity = 0
            reward = self.get_reward(self.eta, 0, fitted_c_mse)
            expr_reward_mse_complexity_tup_ls = [
                (fitted_c_expr, reward, fitted_c_mse, complexity)
            ]
            flag = self.pareto_update_and_check(expr_reward_mse_complexity_tup_ls)
            if flag:
                # Stop condition reached
                return True, self.pareto_frontier

            from sklearn.linear_model import LinearRegression

            reg = LinearRegression().fit(self.X.cpu().numpy(), self.Y.cpu().numpy())
            fitted_linear_expr = str(reg.intercept_[0])
            for i in range(self.n_variables):
                fitted_linear_expr += (
                    " + " + str(reg.coef_[0][i]) + "*" + str(self.variables[i])
                )
            # print(fitted_linear_expr)
            fitted_linear_expr = str(
                prun_constant(sympy.sympify(fitted_linear_expr), self.prun_ndigit)
            )
            # print(fitted_linear_expr)
            fitted_linear_mse = np.mean(
                (self.Y.cpu().numpy() - reg.predict(self.X.cpu().numpy())) ** 2
            )
            # complexity = sympy.count_ops(fitted_linear_expr)
            

            complexity = get_sympy_complexity(fitted_linear_expr)
            reward = self.get_reward(self.eta, 1, fitted_linear_mse)
            expr_reward_mse_complexity_tup_ls = [
                (fitted_linear_expr, reward, fitted_linear_mse, complexity)
            ]
            flag = self.pareto_update_and_check(expr_reward_mse_complexity_tup_ls)
            if flag:
                # Stop condition reached
                return True, self.pareto_frontier

        global_iter = 0
        while time.time() - self.start_time < self.time_limit:
            global_iter += 1

            best_expr, tokens = self.token_generator.step(
                self.n_inputs,
                self.n_sample_variables,
                self.X.cpu().numpy(),
                self.Y.cpu().numpy(),
                use_set=(self.n_variables == self.n_sample_variables),
                reset=False,
                use_float_const=self.use_const,
            )

            sampled_variables = random.sample(self.variables, self.n_sample_variables)

            tokens = sampled_variables + tokens

            expr_ls = tokens
            orginal_X = self.X.cpu().numpy()
            orginal_Y = self.Y.cpu().numpy()
            print("expr_ls", expr_ls)

            sampled_idx = np.unique(
                np.random.choice(
                    orginal_X.shape[0],
                    size=min(self.n_down_sample, orginal_X.shape[0]),
                    replace=False,
                )
            )
            Y = orginal_Y[sampled_idx]
            print("expr_ls", expr_ls, "self.variables", self.variables)
            flag, X = self.get_gs_X(expr_ls, self.variables, orginal_X[sampled_idx])
            print("flag", flag)
            X = X.real
            X = torch.from_numpy(X).to(self.device).float()
            Y = torch.from_numpy(Y).to(self.device).float()

            self.net.current_expr_ls = expr_ls

            expr_best_ls, MSE_min_raw_ls = self.get_best_expr_and_MSE_topk(
                X, Y, self.top_k
            )

            if self.real_time_display:
                if global_iter % self.real_time_display_freq == 0:
                    self.display_expr_table()

            expr_best_ls += [best_expr]
            MSE_min_raw_ls += [0]
            if global_iter > 100:
                try:
                    crossover_exprs = generate_crossover_expressions(
                        [sympy.S(ermc[0]) for ermc in self.pareto_frontier],
                        size=len(expr_best_ls) * 2,
                    )
                    expr_best_ls += crossover_exprs
                    MSE_min_raw_ls += [0] * len(crossover_exprs)
                except Exception:
                    pass

            expr_best_ls, reward_ls, mse_ls, complexity_ls = (
                self.from_expr_MSE_ls_get_ermc_ls(expr_best_ls, MSE_min_raw_ls)
            )
            print("updating pf..")
            for expr_best, reward, mse, complexity in zip(
                expr_best_ls, reward_ls, mse_ls, complexity_ls
            ):
                if type(expr_best) is list:
                    for e, r, m, c in zip(expr_best, reward, mse, complexity):
                        expr_reward_mse_complexity_tup_ls = [(e, r, m, c)]

                        flag = self.pareto_update_and_check(
                            expr_reward_mse_complexity_tup_ls
                        )

                        if flag:
                            # Stop condition reached
                            return True, self.pareto_frontier
                else:
                    expr_reward_mse_complexity_tup_ls = [
                        (expr_best, reward, mse, complexity)
                    ]

                    flag = self.pareto_update_and_check(
                        expr_reward_mse_complexity_tup_ls
                    )

                    if flag:
                        # Stop condition reached
                        return True, self.pareto_frontier

            if len(reward_ls) > 0:
                r = max(reward_ls)
            else:
                r = 0
            print("reward:", r)
            self.token_generator.reward(r, [ermc[0] for ermc in self.pareto_frontier])

        print("updating pf. ok")

        return False, self.pareto_frontier

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        best_expr_str = self.get_pf(sort_by="mse")[0][0]
        print("best_expr_str")
        print(best_expr_str)
        print("self.variables")
        print(self.variables)
        print("X")
        print(X.shape)
        

        Y_pred = expr_to_Y_pred(best_expr_str, X, self.variables)
        return Y_pred

    def replace_variables(self, expr_str, to_repr=True):
        variables, variables_repr = self.variables, self.variables_repr
        if to_repr:
            var_dict = dict(zip(variables, variables_repr))
        else:
            var_dict = dict(zip(variables_repr, variables))

        pattern = r"\b(" + "|".join(re.escape(v) for v in var_dict.keys()) + r")\b"

        def replacer(match):
            return var_dict.get(match.group(1), match.group(1))

        return re.sub(pattern, replacer, expr_str)

    def get_pf(self, sort_by="reward", descend=None):
        dict_ = {
            "expr": (0, False),
            "mse": (2, False),
            "reward": (1, True),
            "complexity": (3, False),
        }
        if descend is None:
            descend = dict_[sort_by][1]
        sort_index = dict_[sort_by][0]

        pareto_frontier = self.pareto_frontier.copy()
        pareto_frontier.sort(key=lambda x: x[sort_index], reverse=descend)

        return pareto_frontier

    def get_params(self):
        return dict(
            variables=self.variables,
            operators=self.operators,
            n_symbol_layers=self.n_symbol_layers,
            n_inputs=self.n_inputs,
            trying_const_range=self.trying_const_range,
            optimizer=self.optimizer,
        )

    def fit_LS(self, expr_str, X, Y, variables, min_MSE_raw, add_bias, together):
        """X,Y: (bs,m), (bs,1) numpy"""

        def get_loss_lm(C):

            functions = {
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                "arcsin": np.arcsin,
                "arccos": np.arccos,
                "arctan": np.arctan,
                "sign": np.sign,
            }

            nonlocal expr_c
            expr_c_temp = expr_c

            for i, c in enumerate(C):
                expr_c_temp = expr_c_temp.replace("C{}".format(i), str(c))

            values = {variables[j]: X[:, j] for j in range(X.shape[1])}
            pred = eval(expr_c_temp.lower(), functions, values)
            true = Y[:, 0]
            diff = true - pred
            square = diff**2
            return np.mean(square)

        # Because of the Piecewise problems in the sympy,
        # a special judgment was made on the sign
        if "sign" in expr_str or not together:
            expr_num = sympy.sympify(se_sympify(expr_str))
        else:
            expr_num = sympy.simplify(expr_str)

        expr_num = set_real(expr_num)
        if expr_num.is_polynomial():
            expr_num = expr_num.expand()

        expr_c, C0 = get_expr_C_and_C0(
            expr_num,
            variables,
            add_bias=add_bias,
            use_replace_exponent=self.use_replace_expo,
        )
        try:
            C0 = np.array(C0).astype(np.float32)
        except:
            return None, np.nan, expr_c, expr_num

        # To prevent repeated fitting of formulas of the same form
        pruned_expr_c = prun_constant(expr_c, n_digits=2)
        if pruned_expr_c not in self.fitted_expr_c_set:
            # self.fitted_expr_c_set.add(expr_c) #
            self.fitted_expr_c_set.add(pruned_expr_c)
        else:
            return None, np.nan, expr_c, expr_num

        expr_c = str(expr_c)

        try:
            result = opt.minimize(get_loss_lm, C0, method=self.optimizer, tol=1e-6)
            if np.isnan(result.fun):
                raise ValueError
        except:
            return None, np.nan, expr_c, expr_num

        best_C = result.x
        final_c = expr_c
        for i, c in enumerate(best_C):
            final_c = final_c.replace("C{}".format(i), str(c))

        return result.x, result.fun, expr_c, final_c

    def remove_one_coeffs(self, expr):
        def traverse(arg):
            if arg.is_Atom:
                if arg in [1.0, -1.0]:
                    return sympy.nsimplify(arg)
                else:
                    return arg
            else:
                return arg.func(*[traverse(a) for a in arg.args])

        return traverse(expr)

    def pareto_update_and_check(self, new_samples):
        index1 = 2  # MSE
        index2 = 3  # Complexity

        for sample in new_samples:
            mse = sample[2]
            expr = sample[0]
            if (
                np.isnan(mse)
                or np.isinf(mse)
                or "nan" in expr
                or "oo" in expr
                or "inf" in expr
            ):
                continue
            if sample[0] in [x[0] for x in self.pareto_frontier]:
                continue
            

            try:
                with time_limit(1, "sleep"):
                    expr_sympy = self.set_real(
                        self.remove_one_coeffs(sympy.S(expr)), self.is_positive
                    )
                    

                    if has_nested_func(expr_sympy):
                        continue
                    expr = str(expr_sympy)
                    if has_irregular_power(expr):
                        continue
            except TimeoutException:
                continue

            sample = (expr, sample[1], sample[2], sample[3])

            self.pareto_frontier.append(sample)

            i = 0
            while i < len(self.pareto_frontier):
                j = i + 1
                while j < len(self.pareto_frontier):
                    if (
                        self.pareto_frontier[i][index1]
                        >= self.pareto_frontier[j][index1]
                        and self.pareto_frontier[i][index2]
                        >= self.pareto_frontier[j][index2]
                    ):
                        # The i-th sample is dominated by the j-th sample, so remove it
                        self.pareto_frontier.pop(i)
                        i -= 1
                        break
                    elif (
                        self.use_strict_pareto
                        and self.pareto_frontier[j][index1]
                        >= self.pareto_frontier[i][index1]
                        and self.pareto_frontier[j][index2]
                        >= self.pareto_frontier[i][index2]
                    ) or (
                        not self.use_strict_pareto
                        and self.pareto_frontier[j][index1]
                        > self.pareto_frontier[i][index1]
                        and self.pareto_frontier[j][index2]
                        > self.pareto_frontier[i][index2]
                    ):

                        # The j-th sample is dominated by the i-th sample, so remove it
                        self.pareto_frontier.pop(j)
                        j -= 1
                    j += 1
                i += 1

            if sample in self.pareto_frontier:
                is_terminate = self.pareto_check(sample)
                if is_terminate:
                    return True
        return False

    def pareto_check(self, sample):
        expr, reward, mse, complexity = sample
        # print('checking ',expr)

        if self.use_threshold and mse < self.threshold:
            print("=" * 40)
            print("Algo. stop, because MSE < threshold")
            print("=" * 40)
            return True
        # elif (self.probe is not None) and self.my_equals(self.my_simplify(expr_best), self.probe):
        elif (self.probe is not None) and self.my_equals(expr, self.probe_evalf):
            print("=" * 40)
            print("Algo. stop, because expr_best == probe")
            print("MSE", mse)
            print("=" * 40)
            return True
        else:
            return False

    def set_real(self, expr_sympy, positive=False):
        for v in expr_sympy.free_symbols:
            expr_sympy = expr_sympy.subs(
                v, sympy.Symbol(str(v), positive=positive, real=True)
            )
        return expr_sympy

    def my_equals(self, expr, probe_evalf):
        

        try:
            with time_limit(1, "sleep"):
                expr = self.set_real(
                    sympy.sympify(self.remove_one_coeffs(sympy.S(expr))),
                    self.is_positive,
                )
                # expr = self.set_real(sympy.sympify(self.remove_one_coeffs(str(expr))), self.is_positive)
                is_equal = (expr.evalf()).equals(probe_evalf) or str(
                    expr.evalf()
                ) == str(probe_evalf)
                return is_equal
        except Exception as e:
            print("error in my_equals", e)
            return False

    def my_simplify(self, expr_str, use_together):
        expr_sympy = se_sympify(expr_str)
        expr_sympy = set_real(expr_sympy)
        if use_together:
            return sympy.cancel(sympy.together(expr_sympy))
        else:
            return expr_sympy

    def display_expr_table(self, sort_by="reward", descend=None, use_repr=True):
        dict_ = {
            "expr": (0, False),
            "mse": (2, False),
            "reward": (1, True),
            "complexity": (3, False),
        }
        if descend is None:
            descend = dict_[sort_by][1]
        sort_index = dict_[sort_by][0]

        pareto_frontier = self.pareto_frontier.copy()

        if use_repr:
            for i in range(len(pareto_frontier)):
                expr, reward, mse, complexity = pareto_frontier[i]
                expr_repr = self.replace_variables(expr, to_repr=True)
                pareto_frontier[i] = (expr_repr, reward, mse, complexity)

        pareto_frontier.sort(key=lambda x: x[sort_index], reverse=descend)

        print("=" * 73)
        print(
            "|",
            "MSE".center(10),
            "|",
            "Complexity".center(10),
            "|",
            "Reward".center(10),
            "|",
            "Expression".center(30),
            "|",
        )
        for i, (expr, reward, mse, complexity) in enumerate(
            pareto_frontier[: self.real_time_display_ntop]
        ):
            print(
                "|",
                format(mse, "10.3e"),
                "|",
                str(complexity).ljust(10),
                "|",
                format(reward, "10.3e"),
                "|",
                expr.ljust(30),
                "|",
            )
        print("=" * 73)

        return pareto_frontier

    def get_reward(self, eta, complexity, mse):
        return (eta**complexity) / (1 + math.sqrt(mse))

    def get_best_expr_and_MSE(self, X, Y):

        with torch.no_grad():
            sum_ = torch.zeros((1, self.net.out_dim), device=self.net.device)
            for i in range(X.shape[0]):
                H = self.net.forward(X[i].reshape(1, -1))
                diff = H - Y[i]
                square = diff**2
                sum_ += square
            mean = sum_ / X.shape[0]
            mean = mean.reshape(-1)

            # replace all nan, -inf to inf
            mean[torch.isnan(mean)] = float("inf")
            mean[torch.isinf(mean)] = float("inf")

            min_value, min_index = torch.min(mean, dim=0)

            MSE_min = min_value.item()
            MSE_mean = torch.mean(mean).item()

            expr_best = self.net.get_expr(round(min_index.item()))

            return expr_best, MSE_min, MSE_mean

    def get_best_expr_and_MSE_topk(self, X, Y, n_top):

        self.fitted_expr_c_set = set()

        with torch.no_grad():
            sum_ = torch.zeros((1, self.net.out_dim), device=self.net.device)
            for i in range(X.shape[0]):
                H = self.net.forward(X[i].reshape(1, -1))
                diff = H - Y[i]
                square = diff**2
                sum_ += square
            mean = sum_ / X.shape[0]
            mean = mean.reshape(-1)

            # replace all nan, -inf to inf
            mean[torch.isnan(mean)] = float("inf")
            mean[torch.isinf(mean)] = float("inf")

            values, indices = torch.topk(mean, n_top, largest=False, sorted=True)
            MSE_min_ls = values.tolist()

            expr_best_ls = []
            from tqdm import tqdm

            for i in tqdm(indices.tolist()):
                expr_best_ls.append(self.net.get_expr(round(i)))
            print("expr_best_ls:")
            print("-" * 20)
            for expr in expr_best_ls:
                print(expr)
            print("-" * 20)
            return expr_best_ls, MSE_min_ls

    def get_gs_X(self, g_list, variables, X):

        # from utils.data import expr_to_Y_pred
        

        """get the base expressions' data (gs)

        Args:
            g_list (list): _description_
            variables (list): _description_
            X (np.ndarray): _description_

        Returns:
            Tuple[bool, np.ndarray]: success flag and gs_X array (n, n_gs), where n_gs == len(g_list)
        """
        success = False

        gs_X = []
        for g in g_list:
            try:
                g_sympy = se_sympify(g)
            except RuntimeError:
                g_sympy = sympy.S("1")
            g_X = expr_to_Y_pred(g_sympy, X, variables)  # -> [n, 1]
            gs_X.append(g_X)

        gs_X = np.hstack(gs_X)
        # keep safe, np.nan or np.inf -> 0
        gs_X[np.isnan(gs_X)] = 0
        gs_X[np.isinf(gs_X)] = 0
        return success, gs_X

    def from_expr_MSE_ls_get_ermc_ls(self, expr_best_ls, MSE_min_raw_ls):
        """
        -> e_ls, r_ls, m_ls, c_ls
        """
        e_ls = []
        r_ls = []
        m_ls = []
        c_ls = []

        reward_max = -1
        

        for expr_best, MSE_min_raw in zip(expr_best_ls, MSE_min_raw_ls):
            try:
                expr_sim = self.my_simplify(expr_best, self.together)
                expr_sim_str = str(expr_sim)

                orginal_X = self.X.cpu().numpy()
                orginal_Y = self.Y.cpu().numpy()

                if not ("nan" in expr_sim_str or "oo" in expr_sim_str):
                    MSE_min_raw = recal_MSE(
                        expr_sim_str, orginal_X, orginal_Y, self.variables
                    )
                if np.isnan(MSE_min_raw) or np.isinf(MSE_min_raw):
                    print("isnan", expr_sim_str)
                    continue
                if has_nested_func(expr_sim):
                    continue
                if has_irregular_power(expr_sim_str):
                    continue
                if "nan" in expr_sim_str or "oo" in expr_sim_str:
                    print("isnan", expr_sim_str)
                    continue
                else:
                    if self.use_const:
                        best_C, MSE_min, expr_c, final_c = self.fit_LS(
                            expr_sim_str,
                            orginal_X,
                            orginal_Y,
                            self.variables,
                            MSE_min_raw,
                            add_bias=self.add_bias,
                            together=self.together,
                        )
                        if best_C is None:
                            continue

                        if self.prun_const:
                            try:
                                final_c = sympy.sympify(se_sympify(str(final_c)))
                                final_c = prun_constant(final_c, self.prun_ndigit)
                            except Exception as e:
                                print("prun_constant error", e)
                                pass

                        expr_best = str(final_c)

                        print(
                            str(expr_sim).ljust(15),
                            "->",
                            expr_c.ljust(15),
                            "-> ",
                            str(final_c).ljust(15),
                        )
                    else:

                        if self.prun_const:
                            expr_sim = self.remove_one_coeffs(
                                prun_constant(expr_sim, self.prun_ndigit)
                            )

                        MSE_min = MSE_min_raw
                        expr_best = str(expr_sim)
                        print("expr_best ", expr_best)
                    

                    complexity = get_sympy_complexity(expr_best)
                    if complexity > 50:
                        continue
                    reward = self.get_reward(self.eta, complexity, MSE_min)

                if reward > reward_max:
                    reward_max = reward

                e_ls.append(expr_best)
                r_ls.append(reward)
                m_ls.append(MSE_min)
                c_ls.append(complexity)

            except RuntimeError:
                print("RuntimeError")
                continue

        return e_ls, r_ls, m_ls, c_ls
