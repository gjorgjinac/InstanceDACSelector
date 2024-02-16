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
for file in [
             "CMA-ES/processed_data.csv",
            ]:

    print('Setting seed')
    set_random_seed(run)

    whole_df=pd.read_csv(f'data/{file}',sep='\t').set_index(['episode','seed','instance'])
    print(whole_df.columns)
    for seed in whole_df.reset_index()['seed'].unique():
        for rai in ['RA','A','R']:
            df=whole_df.query('seed==@seed')
            if rai=='R':
                df=df[list(filter(lambda x: x.startswith('R_'), df.columns))]
            if rai=='A':
                df=df[list(filter(lambda x: x.startswith('A_'), df.columns))]

            if rai=='RA':
                df=df[list(filter(lambda x: x.startswith('A_') or x.startswith('R_'), df.columns))]
            print(rai)
            print(df.columns)
            df.index=[f'eposide_{e}_instance_{i}' for (e,s,i) in df.index]
            df.index.name='i1'
            
            if True or not os.path.isfile(sim_file):
                
                print('Calculating similarity')
                sim=cosine_similarity(df.values,df.values)
                sim=pd.DataFrame(sim, index=df.index,columns=df.index)
                print('Melting similarity')
                sim_melted=sim.reset_index().melt('i1', sim.columns, var_name='i2', value_name='sim')
                #sim_melted.to_csv(sim_file, compression='zip')
            else:
                sim_melted=pd.read_csv(sim_file, compression='zip')
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
                    result_directory=os.path.join("results",file.split("/")[0],algorithm_name,  "Catch22", rai, str(min_similarity_threshold), str(seed) )
                    os.makedirs(result_directory, exist_ok=True)
                    result_file_name = os.path.join(result_directory, f'seed_{seed}_selector_run_{run}.csv')
                    pd.DataFrame(algorithm_results).to_csv(result_file_name)