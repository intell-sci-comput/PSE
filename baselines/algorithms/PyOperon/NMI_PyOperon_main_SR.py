import os
import time
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter

from sympy import parse_expr
import matplotlib.pyplot as plt
from copy import deepcopy

from utils.data import get_benchmark_data

def if_is_exist(path,name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                return True
    return False

# random_seed = 0
random_seed = 19
n_runs = 1
max_time = 3600
seed = random_seed
benchmark_file_ls = ['benchmark.csv','benchmark_Feynman.csv']

hp = {
    'n_runs':n_runs,
    'max_time':max_time,
    'seed':seed,
    }

experiment_name = 'PyOperon_bench'

for key, value in hp.items():
    experiment_name += '_{}'.format(value)

path_log = './log/' + experiment_name + '/'

for benchmark_file in benchmark_file_ls:

    df = pd.read_csv('../../../benchmark/{}'.format(benchmark_file))

    df_save_all = pd.DataFrame(columns=['name', 'recovery_rate', 'avg_time_cost','n_runs'])

    for benchmark_name in df['name']:

        print('Runing benchmark: {}'.format(benchmark_name))

        cnt_success = 0
        sum_time = 0
        
        print('n_runs: {}'.format(n_runs))

        df_save = pd.DataFrame(columns=['name', 'success', 'time_cost',
                                        'expr_str', 'expr_sympy', 'R2', 'MSE', 'reward', 'complexity'])

        for i in range(n_runs):
            np.random.seed(random_seed + i)
            print('Runing {}-th time'.format(i+1))
            
            log_path = './log/PyOperon/benchmark/{}/{}/'.format(experiment_name,benchmark_name)
            os.makedirs(log_path, exist_ok=True)
            csv_name = '{}.csv'.format(random_seed + i)
            log_path_csv = log_path + csv_name
            
            if if_is_exist(log_path, csv_name):
                continue
            
            X, Y, use_constant, expression, variables_name = get_benchmark_data(benchmark_file,
                                    benchmark_name,
                                    1000)

            Input = X
            Output = Y

            np.random.seed(random_seed + i)
            
            start_time = time.time()

            X_train, y_train = X.copy(), Y.copy().flatten()
            X_test, y_test = X.copy(), Y.copy().flatten()

            y_pred = RandomForestRegressor(n_estimators=100).fit(X_train, y_train).predict(X_train)
            sErr = np.sqrt(mean_squared_error(y_train,  y_pred))

            reg = SymbolicRegressor(
                    allowed_symbols= "add,sub,mul,div,sin,cos,exp,logabs,variable"+(",constant" if use_constant else ""),
                    brood_size= 10,
                    comparison_factor= 0,
                    crossover_internal_probability= 0.9,
                    crossover_probability= 1.0,
                    epsilon= 1e-05,
                    female_selector= "tournament",
                    # generations= 1000,
                    generations= 10000000000000, # 
                    initialization_max_depth= 5,
                    initialization_max_length= 10,
                    initialization_method= "btc",
                    irregularity_bias= 0.0,
                    local_search_probability=1.0,
                    lamarckian_probability=1.0,
                    optimizer_iterations=1,
                    optimizer='lm',
                    male_selector= "tournament",
                    max_depth= 10,
                    # max_evaluations= 1000000,
                    max_evaluations= 100000000000000,
                    max_length= 50,
                    max_selection_pressure= 100,
                    model_selection_criterion= "minimum_description_length",
                    mutation_probability= 0.25,
                    n_threads= 4,
                    objectives= [ 'mse', 'length' ],
                    offspring_generator= "os",
                    pool_size= 1000,
                    population_size= 1000,
                    random_state= None,
                    reinserter= "keep-best",
                    # max_time= 900,
                    max_time= max_time,
                    tournament_size=3,
                    uncertainty= [sErr],
                    add_model_intercept_term=use_constant,
                    add_model_scale_term=use_constant,
                    symbolic_mode=not use_constant
                    )

            reg.fit(X_train, y_train)
            res = [(s['objective_values'], s['tree'], s['minimum_description_length']) for s in reg.pareto_front_]
            
            end_time = time.time()
            time_cost = end_time - start_time
            print('time_cost',time_cost)
            
            with open(log_path+'time.txt','a') as f:
                f.write(str(time_cost)+'\n')
            
            results_data = []
            for obj, expr, mdl in res:
                expr_str = reg.get_model_string(expr, 12)
                complexity = obj[1]
                loss = obj[0]
                
                results_data.append({
                    'Complexity': int(complexity),
                    'Loss': loss,
                    'Equation': expr_str
                })
            
            results_data.sort(key=lambda x: x['Complexity'])
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(log_path_csv, index=False)
            
            print("Results saved to:", log_path_csv)
            print("obj,                         expr,                         mdl")
            for obj, expr, mdl in res:
                print(f'{obj}, {mdl:.2f}, {reg.get_model_string(expr, 12)}')
