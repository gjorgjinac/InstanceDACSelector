import os
import random

import numpy as np
import pandas as pd
from networkx import Graph, write_adjlist
from networkx.algorithms.dominating import dominating_set
from networkx.algorithms.mis import maximal_independent_set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import sys

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


import argparse

parser = argparse.ArgumentParser(description="Run selector")

parser.add_argument("--run", type=int, help="Run number")
args = parser.parse_args()

run = args.run
#"sigmoid_2D3M_test_catch22.csv", "sigmoid_2D3M_test.csv", 
for file in ["Sigmoid/Catch 22 rewards_RI/sigmoid_2D3M_train_rewards_catch22_seed_1.csv",
             "Sigmoid/Catch 22 rewards_RI/sigmoid_2D3M_train_rewards_catch22_seed_2.csv",
             "Sigmoid/Catch 22 rewards_RI/sigmoid_2D3M_train_rewards_catch22_seed_3.csv",
             "Sigmoid/Catch 22 rewards_RI/sigmoid_2D3M_train_rewards_catch22_seed_10.csv",
             "Sigmoid/Raw rewards_RI/sigmoid_2D3M_train_raw_seed_1.csv",
             "Sigmoid/Raw rewards_RI/sigmoid_2D3M_train_raw_seed_2.csv",
             "Sigmoid/Raw rewards_RI/sigmoid_2D3M_train_raw_seed_3.csv",
             "Sigmoid/Raw rewards_RI/sigmoid_2D3M_train_raw_seed_10.csv"
            ]:
    for use_params in [True, False]:
    
        print('Setting seed')
        set_random_seed(run)
        param_columns=['state_Shift..dimension.1.', 'state_Shift..dimension.2.','state_Slope..dimension.1.', 'state_Slope..dimension.2.']
        
        df=pd.read_csv(f'data/{file}',sep='\t').set_index(['instance'])
        
        seed=int(file.split('_')[-1].replace('.csv',''))
        print("Seed ", seed)
        if not use_params:
            df=df.drop(param_columns, axis=1)
        df.index=[f'instance_{i}' for i in df.index]
        df.index.name='i1'
        print(df.columns)

        sim=cosine_similarity(df.values,df.values)
        sim=pd.DataFrame(sim, index=df.index,columns=df.index)
        print('Melting similarity')
        sim_melted=sim.reset_index().melt('i1', sim.columns, var_name='i2', value_name='sim')

        print(sim_melted)
        
        
        
        
        for min_similarity_threshold in [0.7,0.8,0.9,0.95]:
            print(min_similarity_threshold)
            g = Graph()
            g.add_nodes_from(df.index)
            print('Adding edges')
            g.add_edges_from(sim_melted.query('sim>@min_similarity_threshold and i1!=i2')[['i1','i2']].values)
    
            for algorithm_name, algorithm_results in [('DS', dominating_set(g)),
                                                      ('MIS', maximal_independent_set(g))]:
                print(len(algorithm_results))
                rai=file.split("/")[1].split("_")[-1] 
                if not use_params:
                    rai=rai.replace("I","")
                result_directory=os.path.join("results",file.split("/")[0],algorithm_name, "Raw" if "raw" in file.lower() else "Catch22", rai, str(min_similarity_threshold), str(seed) )
                os.makedirs(result_directory, exist_ok=True)
                result_file_name = os.path.join(result_directory, f'seed_{seed}_selector_run_{run}.csv')
                pd.DataFrame(algorithm_results).to_csv(result_file_name)
