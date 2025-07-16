import sys

sys.path.append(".")

import os
import click
import time
import numpy as np
import sympy as sp
import pandas as pd


from utils.data import expr_to_Y_pred


@click.command()
@click.option("--experiment_name", default="_", type=str, help="experiment_name")
@click.option("--gpu_index", "-g", default=0, type=int, help="gpu index used")
@click.option(
    "--operators",
    "-l",
    default="['Add','Mul','SemiSub','SemiDiv','Identity']",
    help="operator library",
)
@click.option(
    "--n_inputs",
    "-i",
    default=50,
    type=int,
    help="PSRN input size (n variables + n constants)",
)
@click.option("--seed", "-s", default=0, type=int, help="seed")
@click.option(
    "--topk",
    "-k",
    default=10,
    type=int,
    help="number of best expressions to take from PSRN to fit",
)
@click.option("--time_limit", default=600, type=int, help="time limit (s)")
def main(
    experiment_name,
    gpu_index,
    operators,
    n_inputs,
    seed,
    topk,
    time_limit,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    n_runs = 20

    import torch
    from model.regressor import PSRN_Regressor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print(operators)
    operators = eval(operators)
    print(operators)

    hp = {
        "operators": operators,
        "n_inputs": n_inputs,
        "seed": seed,
    }

    path_log = "./log/many_input/" + experiment_name + "/"

    if not os.path.exists(path_log):
        os.makedirs(path_log)

    cnt_success = 0
    sum_time = 0


    # load many input d50 files
    p_dir = 'data/many_input/synthetic_50d_datasets_with_gt'
    lss = os.listdir(p_dir)
    lss_csv_names = [a for a in lss if a.endswith('csv')]
    lss_txt_names = [a for a in lss if a.endswith('txt')]
    lss_csv_names.sort()
    lss_txt_names.sort()
    print(lss_csv_names)
    print(lss_txt_names)

    df_save_all = pd.DataFrame(
        columns=["name", "recovery_rate", "avg_time_cost", "n_runs"]
    )
    for csv_name, txt_name in zip(lss_csv_names, lss_txt_names):


        df_save = pd.DataFrame(
            columns=[
                "name",
                "success",
                "time_cost",
                "R2",
                "MSE",
                "reward",
                "complexity",
                "expr_str_best_Reward",
                "expr_sympy_best_Reward",
                "expr_str_best_MSE",
                "expr_sympy_best_MSE",
            ]
        )

        for i in range(n_runs):
            p_csv = os.path.join(p_dir, csv_name)
            p_txt = os.path.join(p_dir, txt_name)
            class_name_ls = open(p_txt).readlines()
            lines = [line.strip() for line in class_name_ls]
            gt = lines[0]
            print(gt)
            

            df = pd.read_csv(p_csv, header=None)

            variables_name = [f"x_{str(i+1).zfill(2)}" for i in range(50)]

            target_name = ["y"]

            Input = df.values[:, :-1].reshape(len(df), -1)
            Output = df.values[:, -1].reshape(len(df), 1)

            Input = torch.from_numpy(Input).to(device).to(torch.float32)
            Output = torch.from_numpy(Output).to(device).to(torch.float32)

            print(Input.shape, Output.shape)
            print(Input.dtype, Output.dtype)
        

            regressor = PSRN_Regressor(
                variables=variables_name,
                n_symbol_layers=2,
                dr_mask_dir="./dr_mask",
                use_const=False,
                device="cuda",
                token_generator='GP',
                stage_config={
                    "default": {
                        "operators": operators,
                        "time_limit": time_limit,
                        "n_psrn_inputs": n_inputs,
                        "n_sample_variables": n_inputs-10,
                    },
                    "stages": [
                        {},
                    ],
                },
            )
            
            regressor.config['GP']['base']['tokens'] = ['Add', 'Sub', 'Mul', 'Div']
            regressor.config['GP']['base']['has_const'] = False
            print(regressor.config['GP']['base']['tokens'])
            print(regressor.config['GP']['base']['has_const'])
            
            

            start = time.time()
            flag, pareto_ls = regressor.fit(
                Input,
                Output,
                use_threshold=True,
                threshold=1e-12,
                probe=gt,  # expression probe, string, stop if probe in pf
                prun_const=True,
                prun_ndigit=2,
                n_down_sample=100,
                top_k=topk,
            )
            end = time.time()
            time_cost = end - start

            crits = ["reward", "mse"]

            for crit in crits:
                print("Pareto Front sort by {}".format(crit))
                pareto_ls = regressor.display_expr_table(sort_by=crit)

            expr_str, reward, loss, complexity = pareto_ls[0]
            expr_sympy = sp.simplify(expr_str)
            ############# Print Pareto Front ###############
            crits = ["mse", "reward"]

            expr_str_best_reward = None
            expr_sympy_best_reward = None
            expr_str_best_MSE = None
            expr_sympy_best_MSE = None

            for crit in crits:
                print("Pareto Front sort by {}".format(crit))
                pareto_ls = regressor.display_expr_table(sort_by=crit)
                expr_str, reward, loss, complexity = pareto_ls[0]

                if crit == "mse":
                    expr_str_best_MSE = expr_str
                    expr_sympy_best_MSE = sp.simplify(expr_str)
                else:
                    expr_str_best_reward = expr_str
                    expr_sympy_best_reward = sp.simplify(expr_str)
            print(expr_str)

            print("time_cost", time_cost)
            if flag:
                print("[*** Found Expr ! ***]")
                cnt_success += 1
            sum_time += time_cost



            df_save = df_save.append(
                {
                    "name": csv_name,
                    "success": flag,
                    "time_cost": time_cost,
                    "expr_str_best_Reward": expr_str_best_reward,
                    "expr_sympy_best_Reward": expr_sympy_best_reward,
                    "expr_str_best_MSE": expr_str_best_MSE,
                    "expr_sympy_best_MSE": expr_sympy_best_MSE,
                    "MSE": loss,
                    "reward": reward,
                    "complexity": complexity,
                },
                ignore_index=True,
            )


            print(expr_sympy)
            df_hof = pd.DataFrame(
                pareto_ls, columns=["expr_str", "reward", "MSE", "complexity"]
            )
            df_hof = df_hof.head(20)
            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

            if not os.path.exists(path_log + "/{}".format(csv_name)):
                os.makedirs(path_log + "/{}".format(csv_name))
            df_hof.to_csv(
                path_log + "{}/pf_{}_{}.csv".format(csv_name, i, t),
                index=False,
            )
            
            

        # save df_save
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        df_save.to_csv(
            path_log + "benchmark_{}_{}.csv".format(csv_name, t), index=False
        )

        avg_time = sum_time / n_runs
        avg_success_rate = cnt_success / n_runs
        df_save_all = df_save_all.append(
            {
                "name": csv_name,
                "recovery_rate": avg_success_rate,
                "avg_time_cost": avg_time,
                "n_runs": n_runs,
            },
            ignore_index=True,
        )

if __name__ == "__main__":
    main()
