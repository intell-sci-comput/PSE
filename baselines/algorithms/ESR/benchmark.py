
import os
import time
import numpy as np
import pandas as pd
import sympy as sp
import sys
from mpi4py import MPI

sys.path.insert(0, 'ESR')

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import GaussLikelihood
from esr.fitting.likelihood import CCLikelihood, Likelihood, PoissonLikelihood
from esr.fitting.fit_single import single_function
import esr.plotting.plot

from utils.data import get_benchmark_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def if_is_exist(path, name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                return True
    return False

def ensure_function_library_exists(runname, max_comp):

    for comp in range(1, max_comp + 1):
        lib_path = f'ESR/esr/function_library/{runname}/compl_{comp}/'
        if not os.path.exists(lib_path) or not os.path.exists(f'{lib_path}/unique_equations_{comp}.txt'):
            esr.generation.duplicate_checker.main(runname, comp, track_memory=False)
    
    comm.Barrier() 

def select_function_set(use_constant, base_set='koza_maths'):
    if base_set == 'koza_maths':
        if use_constant:
            return 'koza_maths_const'  # ["x", "a"]
        else:
            return 'koza_maths'        # ["x"]
    elif base_set == 'core_maths':
        return 'core_maths'
    else:
        return base_set

def run_esr_benchmark(comp=5, n_runs=20, base_function_set='koza_maths'):
    random_seed = 0
    ESR_n_iter = 10000
    seed = random_seed
    benchmark_file_ls = ['benchmark.csv', 'benchmark_Feynman.csv']

    hp = {
        'n_runs': n_runs,
        'ESR_n_iter': ESR_n_iter,
        'seed': seed,
        'complexity': comp,
    }

    experiment_name = 'ESR_bench'
    for key, value in hp.items():
        experiment_name += '_{}'.format(value)

    path_log = './log/' + experiment_name + '/'

    used_function_sets = set()

    for benchmark_file in benchmark_file_ls:
        
        bench_path = '../../../benchmark/{}'.format(benchmark_file)
        df = pd.read_csv(bench_path)

        df_save_all = pd.DataFrame(columns=['name', 'recovery_rate', 'avg_time_cost', 'n_runs'])

        for benchmark_name in df['name']:
            
            cnt_success = 0
            sum_time = 0
            
            df_save = pd.DataFrame(columns=['name', 'success', 'time_cost',
                                          'expr_str', 'expr_sympy', 'R2', 'MSE', 'reward', 'complexity'])

            for i in range(n_runs):
                if rank == 0:
                    print(f'    {i+1}/{n_runs} runs (complexity={comp})')
                
                np.random.seed(random_seed + i)
                
                log_path = './log/ESR/benchmark/{}/{}/{}/'.format(experiment_name, benchmark_name, random_seed + i)
                if rank == 0:
                    os.makedirs(log_path, exist_ok=True)
                
                comm.Barrier() 
                
                results_pretty_name = 'results_pretty_{}.txt'.format(comp)
                log_path_csv = log_path + results_pretty_name
                
                if if_is_exist(log_path, results_pretty_name):
                    if rank == 0:
                        print(f"   skipping results: {results_pretty_name}")
                    continue
                
                X, Y, use_constant, expression, variables_name = get_benchmark_data(
                    benchmark_file, benchmark_name, 1000)
                if use_constant:
                    continue

                n_variables = X.shape[1]
                if n_variables != 1:
                    if rank == 0:
                        print(f'   ⚠️  {benchmark_name} this problems is NOT single variable!, ESR not support, skipping')
                    continue
                else:
                    if rank == 0:
                        print(f'   ✓ {benchmark_name} single variable problem')

                function_set = select_function_set(use_constant, base_function_set)
                used_function_sets.add(function_set)
            
                start_time = time.time()
                
                
                ensure_function_library_exists(function_set, comp)
                
                x = X.flatten()
                y = Y.flatten()
                yerr = np.full(x.shape, 1.0)
                
                data_filename = f'data_txts/data_{benchmark_name}_{i}_{rank}.txt'
                np.savetxt(data_filename, np.array([x, y, yerr]).T)
                
                likelihood = GaussLikelihood(data_filename, 'gauss_example', 
                                            fn_set=function_set, data_dir=os.getcwd())
                likelihood.out_dir = log_path

                current_comp = comp

                esr.fitting.test_all.main(current_comp, likelihood)
                esr.fitting.test_all_Fisher.main(current_comp, likelihood)
                esr.fitting.match.main(current_comp, likelihood)
                esr.fitting.combine_DL.main(current_comp, likelihood)
                esr.fitting.plot.main(current_comp, likelihood)

                if os.path.exists(data_filename):
                    os.remove(data_filename)

                #####################################

                end_time = time.time()
                time_cost = end_time - start_time
                
                if rank == 0:
                    print(f'time cost: {time_cost:.2f} seconds')
                    with open(log_path + 'time.txt', 'a') as f:
                        f.write(str(time_cost) + '\n')

                sum_time += time_cost

            if rank == 0:
                avg_time = sum_time / n_runs if n_runs > 0 else 0
                print(f"{benchmark_name} success! {avg_time:.2f} seconds")

def main():
    comp = 10
    n_runs = 20
    base_function_set = 'koza_maths'
    
    if len(sys.argv) > 1:
        comp = int(sys.argv[1])

    if len(sys.argv) > 2:
        n_runs = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        base_function_set = sys.argv[3]

    run_esr_benchmark(comp, n_runs, base_function_set)

if __name__ == "__main__":
    main()


