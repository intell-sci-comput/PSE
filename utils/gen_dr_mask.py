import sys

sys.path.append(".")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set gpu idx here for generating drmask

import numpy as np
import sympy
from tqdm import tqdm
from model.models import PSRN
import click


@click.command()
@click.option("--n_symbol_layers", type=int, help="Number of Symbol Layers.")
@click.option("--n_inputs", type=int, help="Number of PSRN Inputs.")
@click.option(
    "--ops",
    help="`basic` or `koza` or Operators List e.g. ['Add','Mul','Identity',...]",
)
@click.option("--save_dir", type=str, default="./dr_mask", help="Mask Save Dir")
def generate_dr_mask(n_symbol_layers, n_inputs, ops, save_dir):
    """
    generate DR mask of PSRN's DRLayer

    Example
    =======
    >>> python utils/gen_dr_mask.py --n_symbol_layers=3 --n_inputs=5 --ops=koza

    Returns:
        None : save mask to dir `./dr_mask`
    """

    if ops == "basic":
        ops = ["Add", "Mul", "Identity", "Neg", "Inv", "Sin", "Cos", "Exp", "Log"]
    elif ops == "koza":
        ops = [
            "Add",
            "Mul",
            "Sub",
            "Div",
            "Identity",
            "Neg",
            "Inv",
            "Sin",
            "Cos",
            "Exp",
            "Log",
        ]
    elif ops == "basic_sign":
        ops = ["Add", "Mul", "Identity", "Neg", "Inv", "Sign"]
    elif ops == "koza_sign":
        ops = ["Add", "Mul", "Sub", "Div", "Identity", "Sign"]
    else:
        print(ops)
        ops = eval(ops)
        print(ops)
        assert type(ops) == list

    input_variable_names = ["x{}".format(i + 1) for i in range(n_inputs)]

    n_layers = n_symbol_layers - 1
    qsrn = PSRN(
        n_variables=len(input_variable_names),
        operators=ops,
        n_symbol_layers=n_layers,
        device="cuda",
    )

    qsrn.current_expr_ls = input_variable_names

    out_expr_ls = []

    print("generating expressions ...")
    for out_index in tqdm(range(qsrn.out_dim)):
        expr = qsrn.get_expr(out_index)
        out_expr_ls.append(expr)

    print("sympifying ...")
    out_expr_sympy_ls = []
    for expr_str in tqdm(out_expr_ls):
        expr_sympy = sympy.sympify(expr_str)
        out_expr_sympy_ls.append(expr_sympy)

    out_expr_sympy_hash_ls = [hash(expr) for expr in out_expr_sympy_ls]

    def get_mask_ls():
        mask_ls = []
        select_expr_hash_ls = []
        for i in range(len(out_expr_sympy_hash_ls)):
            expr_hash = out_expr_sympy_hash_ls[i]

            if expr_hash not in select_expr_hash_ls:
                select_expr_hash_ls.append(expr_hash)
                mask_ls.append(True)
            else:
                mask_ls.append(False)
        return mask_ls

    print("removing duplicate expressions ...")
    mask_ls = get_mask_ls()
    # to numpy bool array
    mask_np = np.array(mask_ls)
    print("Final Expressions", mask_np.sum())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f'{n_layers + 1}_{len(input_variable_names)}_[{"_".join(ops)}]'
    # save
    p = f"{save_dir}/{file_name}_mask.npy"
    np.save(p, mask_np)
    print("Saved >>> {} <<<".format(p))


if __name__ == "__main__":
    generate_dr_mask()

