import os
import time
import numpy as np
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

experiment_name = "PyOperon_many_input"
time_limit = 600  
n_runs = 20 

hp = {
    'time_limit': time_limit,
}

for key, value in hp.items():
    experiment_name += f"_{value}"

path_log = f"./log/pyoperon_many_input/{experiment_name}/"
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
        print(f"Running benchmark for: {csv_name}, seed: {seed}")

        p_csv = os.path.join(p_dir, csv_name)
        df = pd.read_csv(p_csv, header=None)

        variables_name = [f"x_{str(i+1).zfill(2)}" for i in range(50)]
        
        X = df.values[:, :-1]
        y = df.values[:, -1]

        p_txt = os.path.join(p_dir, txt_name)
        with open(p_txt, 'r') as f:
            ground_truth_expr = f.readline().strip()
        print(f"Ground Truth: {ground_truth_expr}")

        log_path_csv = os.path.join(path_log, csv_name.replace('.csv', f'_pf_{seed}.csv'))
        log_path_time = os.path.join(path_log, 'time.txt')
        
        if os.path.exists(log_path_csv):
            print(f'exist {log_path_csv}, skip.')
            continue
        
        # Calculate uncertainty using RandomForest
        y_pred = RandomForestRegressor(n_estimators=100, random_state=seed).fit(X, y).predict(X)
        sErr = np.sqrt(mean_squared_error(y, y_pred))

        use_constant = False
        reg = SymbolicRegressor(
            allowed_symbols="add,sub,mul,div,constant,variable",  
            brood_size=10,
            comparison_factor=0,
            crossover_internal_probability=0.9,
            crossover_probability=1.0,
            epsilon=1e-05,
            female_selector="tournament",
            generations=10000000000000,
            initialization_max_depth=5,
            initialization_max_length=10,
            initialization_method="btc",
            irregularity_bias=0.0,
            local_search_probability=1.0,
            lamarckian_probability=1.0,
            optimizer_iterations=1,
            optimizer='lm',
            male_selector="tournament",
            max_depth=10,
            max_evaluations=100000000000000,
            max_length=50,
            max_selection_pressure=100,
            model_selection_criterion="minimum_description_length",
            mutation_probability=0.25,
            n_threads=4,
            objectives=['mse', 'length'],
            offspring_generator="os",
            pool_size=1000,
            population_size=1000,
            random_state=seed,
            reinserter="keep-best",
            max_time=time_limit,
            tournament_size=3,
            uncertainty=[sErr],
            add_model_intercept_term=use_constant,
            add_model_scale_term=use_constant,
            symbolic_mode=not use_constant
        )
        
        start_time = time.time()
        reg.fit(X, y)
        end_time = time.time()
        time_cost = end_time - start_time
    
        with open(log_path_time, 'a') as f:
            f.write(f"{time_cost}\n")
        
        res = [(s['objective_values'], s['tree'], s['minimum_description_length']) 
               for s in reg.pareto_front_]
        
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
        
        print(f"\n--- Results for {csv_name}, seed {seed} ---")
        print(f"Time cost: {time_cost:.2f} seconds")
        print(f"Pareto front size: {len(results_data)}")
        if results_data:
            best_result = min(results_data, key=lambda x: x['Loss'])
            print(f"Best equation: {best_result['Equation']}")
            print(f"Best loss: {best_result['Loss']}")
            print(f"Best complexity: {best_result['Complexity']}")
        print("="*60 + "\n")

print("All benchmarks finished.")
