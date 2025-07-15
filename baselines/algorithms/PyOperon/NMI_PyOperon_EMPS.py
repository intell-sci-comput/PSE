import os
import time
import numpy as np
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils.data import get_dynamic_data

p = './log/EMPS/PyOperon/'
os.makedirs(p, exist_ok=True)

n_seeds = 20
TIMEOUT = 90

for seed in range(n_seeds):
    print(f"Running seed {seed}")
    
    df, variables_name, target_name = get_dynamic_data('emps','emps')
    # select half of the data as train set
    df = df.iloc[:len(df)//2,:]
    
    np.random.seed(seed)
    
    Input = df[variables_name].values
    Output = df[target_name].values.reshape(-1,1)
    
    if os.path.exists(p+'pf_{}.csv'.format(seed)):
        print('exist {}, skip.'.format(p+'pf_{}.csv').format(seed))
        continue
    
    # Calculate uncertainty using RandomForest
    y_pred = RandomForestRegressor(n_estimators=100).fit(Input, Output.flatten()).predict(Input)
    sErr = np.sqrt(mean_squared_error(Output.flatten(), y_pred))

    reg = SymbolicRegressor(
        allowed_symbols="add,sub,mul,div,sin,cos,exp,logabs,tanh,cosh,abs,constant,variable",
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
        max_time=TIMEOUT,
        tournament_size=3,
        uncertainty=[sErr],
        add_model_intercept_term=True,
        add_model_scale_term=True
    )
    
    start_time = time.time()
    reg.fit(Input, Output.flatten())
    end_time = time.time()
    time_cost = end_time - start_time
    
    with open(p + 'time.txt', 'a') as f:
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
    results_df.to_csv(p + f'pf_{seed}.csv', index=False)
    
    print(f"Results for seed {seed} saved to {p}pf_{seed}.csv")
    print(f"Time cost: {time_cost:.2f} seconds")
