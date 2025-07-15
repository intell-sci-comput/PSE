import os 
import gc
import time
import numpy as np
import pandas as pd
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils.data import add_noise

class_name_ls = open('../../../data/dystsnames/dysts_16_flows.txt').readlines()
class_name_ls = [line.strip() for line in class_name_ls]

print(class_name_ls)

sample_size = 1000
noise_level = 0.1
pts_per_period = 100
n_seeds = 50
TIMEOUT = 60*10  # 10 minutes

for seed in range(50):
    
    experi_name = 'chaotic'
    
    for class_name in class_name_ls:
            
        dysts_model = eval(class_name)
        dysts_model = dysts_model()
        if class_name == 'ForcedBrusselator':
            dysts_model.f = 0
        elif class_name == 'Duffing':
            dysts_model.gamma = 0
        elif class_name == 'ForcedVanDerPol':
            dysts_model.a = 0
        elif class_name == 'ForcedFitzHughNagumo':
            dysts_model.f = 0
        
        np.random.seed(seed)

        print(dysts_model)
        t, data = dysts_model.make_trajectory(sample_size,
                                                return_times=True,
                                                pts_per_period=pts_per_period,
                                                resample=True,
                                                noise=0,
                                                postprocess=False)
        
        # add Gaussian noise
        for k in range(data.shape[1]):
            data[:,k:k+1] = add_noise(data[:,k:k+1], noise_level, seed)
            
        t = t.flatten()
        # t: Nx1 ; sol:NxD
        dim = data.shape[1]
        print('dim',dim)
        if dim == 3:
            vars = ['x','y','z']
            dotsvarnames = ['xdot','ydot','zdot']
            deriv_idxs = [0,1,2]
            trying_const_num = 2
            n_inputs = 5
        elif dim == 4:
            vars = ['x','y','z','w']
            dotsvarnames = ['xdot','ydot','zdot','wdot']
            deriv_idxs = [0,1,2,3]
            trying_const_num = 1
            n_inputs = 5
        else:
            continue
        
        sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
        data_deriv = np.zeros((data.shape[0],len(deriv_idxs)))
        for i,idx in enumerate(deriv_idxs):
            print(data[:,idx:idx+1].shape,t.shape)
            deriv_data_i = sfd._differentiate(data[:,idx:idx+1], t)
            data_deriv[:,i:i+1] = deriv_data_i
        data_all = np.hstack([data, data_deriv])
        
        for idxdot, vardotname in enumerate(dotsvarnames):
            print(f"Running seed {seed}, class {class_name}, variable {vardotname}")
            
            hp_str = '{}/{}/{}/{}'.format(noise_level, experi_name, class_name, vardotname)
            p = './log/dysts/pyoperon/{}/'.format(hp_str)
            os.makedirs(p, exist_ok=True)
                
            if os.path.exists(p+'pf_{}.csv'.format(seed)):
                print('exist {}, skip.'.format(p+'pf_{}.csv'.format(seed)))
                continue
            
            Input = data
            Output = data_deriv[:,idxdot:idxdot+1]
            
            np.random.seed(seed)
            
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
            
            print(f"Results for seed {seed}, {class_name}, {vardotname} saved")
            print(f"Time cost: {time_cost:.2f} seconds")
