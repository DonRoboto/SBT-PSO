from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import argparse
import pandas as pd
from datetime import datetime

import ray
from ray.tune import run, sample_from
import shutil
from spbt_pso_v1 import SwarmBT_PSOLearning


# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1000000)
    
    parser.add_argument("--algo", type=str, default='PPO')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--freq", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000) # make this 1000 for other envs
    #parser.add_argument("--perturb", type=float, default=0.25)
    parser.add_argument("--perturb", type=float, default=0.5)
    
    #parser.add_argument("--env_name", type=str, default="BipedalWalker-v3")
    #parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v3")
    #parser.add_argument("--env_name", type=str, default="Ant-v5")
    #parser.add_argument("--env_name", type=str, default="HalfCheetah-v5")
    #parser.add_argument("--env_name", type=str, default="Swimmer-v5")
    #parser.add_argument("--env_name", type=str, default="Walker2d-v5")
    parser.add_argument("--env_name", type=str, default="InvertedDoublePendulum-v5")
    
    parser.add_argument("--criteria", type=str, default="timesteps_total") # "training_iteration"
    parser.add_argument("--net", type=str, default="32_32")
    parser.add_argument("--batchsize", type=str, default="1000_60000")
    parser.add_argument("--use_lstm", type=int, default=0) # for future, not used
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--method", type=str, default="sbt_pso") 


    args = parser.parse_args()



    timelog = str(datetime.date(datetime.now())) + '_' + str(datetime.time(datetime.now()))

    for seed in range(0, 7):
        
        ray.init()
        
        sbt_pso = SwarmBT_PSOLearning(
            time_attr=args.criteria,
            metric="env_runners/episode_reward_mean",
            mode="max",
            perturbation_interval=args.freq,
            resample_probability=args.perturb,
            quantile_fraction=args.perturb, # copy bottom % with top %
            # Specifies the mutations of these hyperparams
            hyperparam_mutations={
                "lambda": lambda a=0.9, b=1.0: random.uniform(a, b),
                "clip_param": lambda a=0.1, b=0.5: random.uniform(a, b),
                "lr": lambda a=1e-5, b=1e-3: random.uniform(a, b),
                "train_batch_size": lambda a=1000, b=60000: random.randint(a, b),
                

            },
            custom_explore_fn=explore)
        
        methods = {'sbt_pso': sbt_pso
                   }

        
        args.seed = seed

        analysis = run(
            args.algo,
            name="{}_{}_{}_seed{}_{}_{}".format(timelog, args.method, args.env_name, str(args.seed), args.max, args.freq),
            scheduler=methods[args.method],
            #verbose=3,
            verbose=0,
            num_samples= args.num_samples,
            
            checkpoint_at_end=True,
            
            stop= {args.criteria: args.max},
            config= {
                "env": args.env_name,
                #"log_level": "INFO",
                "log_level": "CRITICAL",
                
                "seed": args.seed,
                "kl_coeff": 1.0,
                "num_workers": args.num_workers,
                # "monitor": True, uncomment this for videos... it may slow it down a LOT, but hey :)
                
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 128,

                "num_gpus": 0,
                "horizon": args.horizon,
                "observation_filter": "NoFilter",
                "model": {'fcnet_hiddens': [int(args.net.split('_')[0]) ,int(args.net.split('_')[1])],
                          'free_log_std': True,
                          'use_lstm': args.use_lstm
                          },

                "lambda": sample_from(
                    lambda spec: random.uniform(0.9, 1.0)),
                "clip_param": sample_from(
                    lambda spec: random.uniform(0.1, 0.5)),
                "lr": sample_from(
                    lambda spec: random.uniform(1e-5, 1e-3)),
                "train_batch_size": sample_from(
                    lambda spec: random.randint(1000, 60000)),
 
            }
        )

        all_dfs = analysis.trial_dataframes
        names = list(all_dfs.keys())

        results = pd.DataFrame()
  
        for i in range(args.num_samples):
            df = all_dfs[names[i]]
            
            if df.shape!=(0,0):
                #break
                    
                df = df[['timesteps_total', 'time_total_s','env_runners/num_episodes', 'env_runners/episode_reward_mean', 'info/learner/default_policy/learner_stats/cur_kl_coeff',]]
                df['Agent'] = i
                results = pd.concat([results, df]).reset_index(drop=True)

        args.dir = "{}_{}_{}_Size{}_{}_{}_{}_{}_{}".format(args.algo, args.filename, args.method, str(args.num_samples), args.env_name, args.criteria, args.max, args.freq, args.batchsize)
        exist_dir = os.path.expanduser('./data/' + args.dir)
        if not(os.path.exists(exist_dir)):
            os.makedirs(exist_dir)

        result_dir1 = os.path.expanduser('./data/')
        result_dir2 = "{}/seed{}.csv".format(args.dir, str(args.seed))
        results.to_csv(result_dir1 + "{}/seed{}.csv".format(args.dir, str(args.seed)))
        
        best_trial = analysis.get_best_trial(metric="env_runners/episode_reward_mean", mode="max")
        ckpt = best_trial.checkpoint
        best_checkpoint = ckpt.to_directory()
        
        
        src_dir = best_checkpoint
        dest_dir = "./data/{}/check_{}/".format(args.dir, str(args.seed))
        shutil.copytree(src_dir, dest_dir)

        ray.shutdown()
