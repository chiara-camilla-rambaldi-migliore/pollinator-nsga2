# -*- coding: utf-8 -*-

from pylab import *

from inspyred import benchmarks
from inspyred_utils import NumpyRandomWrapper

import sys

import collections
collections.Iterable = collections.abc.Iterable
collections.Sequence = collections.abc.Sequence
from bumblebee_problem import UrbanPollinator
from multi_objective_bumblebee import run_nsga2
import logging
import time
import os
import pandas as pd
from utils import grayToDecimal
    
""" 
-------------------------------------------------------------------------
Edit this part to do the exercises

"""

display = True# Plot initial and final populations
num_vars = 20 

# parameters for NSGA-2
args = {}
args["pop_size"] = 1
args["max_generations"] = 0

problem = UrbanPollinator()

"""
-------------------------------------------------------------------------
"""

args["fig_title"] = 'NSGA-2'
args["variations_args"] = [
    {"blx_points": [0]},
    {"spx_points": [1, 2, 3, 4, 5, 6, 7, 8]},
    {"spx_points": [9,10,11,12,13,14,15,16]},
    {"spx_points": [17,18,19]},
    {"g_points": [0]},
    {"bf_points": [1, 2, 3, 4, 5, 6, 7, 8]},
    {"bf_points": [9,10,11,12,13,14,15,16]},
    {"bf_points": [17,18,19]},
]
    
if __name__ == "__main__" :
    if len(sys.argv) > 1 :
        rng = NumpyRandomWrapper(int(sys.argv[1]))
    else :
        rng = NumpyRandomWrapper()

    logger = logging.getLogger('inspyred.ec')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('inspyred.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    final_pop, final_pop_fitnesses = run_nsga2(rng, problem,
                                        display=display, num_vars=num_vars,
                                        **args)
    new_final_pop = []
    for i, guy in enumerate(final_pop):
        new_guy = []
        new_guy.append(guy[0])
        new_guy.append(grayToDecimal(guy[1:9]))
        new_guy.append(grayToDecimal(guy[9:17]))
        new_guy.append(grayToDecimal(guy[17:]))
        new_guy.append(final_pop_fitnesses[i][0])
        new_guy.append(final_pop_fitnesses[i][1])
        new_final_pop.append(new_guy)
    
    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)

    df = pd.DataFrame(new_final_pop, columns=['no_mow_pc', 'mowing_days', 'pesticide_days', 'flower_area_type', 'fitness_1', 'fitness_2'])
    df.to_csv('final_pop.csv', index=False)
    print(df)
    
    ioff()
    show()