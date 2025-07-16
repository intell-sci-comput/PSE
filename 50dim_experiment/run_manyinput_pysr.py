import os
import time
import numpy as np
import pandas as pd
from pysr import PySRRegressor

experiment_name = "PySR_many_input"
time_limit = 600  
pysr_n_iter = 1000000000 
n_runs = 20 

hp = {
    'time_limit': time_limit,
    'pysr_n_iter': pysr_n_iter
}

for key, value in hp.items():
    experiment_name += f"_{value}"

path_log = f"./log/pysr_many_input/{experiment_name}/"
if not os.path.exists(path_log):
    os.makedirs(path_log)
    print(f"Created log directory: {path_log}")

p_dir = 'data/many_input/synthetic_50d_datasets_with_gt'

lss = os.listdir(p_dir)
lss_csv_names = sorted([a for a in lss if a.endswith('.csv')])
lss_txt_names = sorted([a for a in lss if a.endswith('.txt')])

print(f"Found {len(lss_csv_names)} problems to run.")
print("---")

for csv_name, txt_name in zip(lss_csv_names, lss_txt_names):
    
    for seed in range(n_runs):
        print(f"Running benchmark for: {csv_name}")

        p_csv = os.path.join(p_dir, csv_name)
        df = pd.read_csv(p_csv, header=None)

        variables_name = [f"x_{str(i+1).zfill(2)}" for i in range(50)]
        
        X = df.values[:, :-1]
        y = df.values[:, -1]

        p_txt = os.path.join(p_dir, txt_name)
        with open(p_txt, 'r') as f:
            ground_truth_expr = f.readline().strip()
        print(f"Ground Truth: {ground_truth_expr}")


        log_path_csv = os.path.join(path_log, csv_name.replace('.csv', f'_equations_{seed}.csv'))
        print('log_path_csv')
        print(log_path_csv)
        
        if os.path.exists(log_path_csv):
            continue
        
        model = PySRRegressor(
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[],
            complexity_of_constants=100, 
            timeout_in_seconds=time_limit,
            niterations=pysr_n_iter,
            deterministic=True,
            random_state=0,
            procs=0,
            multithreading=False,
            equation_file=log_path_csv,
            temp_equation_file=False,
        )

        start_time = time.time()
        model.fit(X, y, variable_names=variables_name)
        end_time = time.time()
        time_cost = end_time - start_time

        print(f"\n--- Results for {csv_name} ---")
        print(f"Time cost: {time_cost:.2f} seconds")
        
        print("Best equation found:")

        best_eq = model.get_best()
        print(f"  Expression: {best_eq['equation']}")
        print(f"  Complexity: {best_eq['complexity']}")
        print(f"  Loss (MSE): {best_eq['loss']}")
        print(f"  Score: {best_eq['score']}")

        print("\nPareto Front:")
        print(model)
        print("="*60 + "\n")

print("All benchmarks finished.")