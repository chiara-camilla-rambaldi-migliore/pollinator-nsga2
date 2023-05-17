# -*- coding: utf-8 -*-

from pylab import *

from inspyred import benchmarks
from inspyred_utils import NumpyRandomWrapper

import sys

import collections
collections.Iterable = collections.abc.Iterable
collections.Sequence = collections.abc.Sequence
from bumblebee_problem_gray_doe import UrbanPollinator
from multi_objective_bumblebee import run_nsga2
import logging
import pandas as pd
from inspyred.ec import variators
from custom_variators import single_point_crossover, bit_flip_mutation, gaussian_mutation
from utils import grayToDecimal
from custom_nsga2 import CustomDoeNSGA2
from inspyred_utils import initial_pop_observer_gray
    
""" 
-------------------------------------------------------------------------
Edit this part to do the exercises

"""

display = False# Plot initial and final populations
num_vars = 20 

# parameters for NSGA-2
args = {}
args["pop_size"] = 10
args["max_generations"] = 30

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
    {"g_points": [0], "gaussian_stdev": 0.1}, # 95% of gaussian values are in the range [-0.2, 0.2]
    {"bf_points": [1, 2, 3, 4, 5, 6, 7, 8]},
    {"bf_points": [9,10,11,12,13,14,15,16]},
    {"bf_points": [17,18,19]},
]

args["max_cores"] = 10
args["seed"] = 23

args["fileName_initial_pop"] = 'pop_gray_doe.csv'
    
if __name__ == "__main__" :
    if len(sys.argv) > 1 :
        rng = NumpyRandomWrapper(int(sys.argv[1]))
        args["seed"] = int(sys.argv[1])
    else :
        rng = NumpyRandomWrapper()

    logger = logging.getLogger('inspyred.ec')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('inspyred_gray_doe.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    variator = [
        variators.blend_crossover, # for no mow percentual
        single_point_crossover, # for mowing days
        single_point_crossover, # for pesticide days
        single_point_crossover, # for flower area type
        gaussian_mutation, # for real values
        bit_flip_mutation, # for binary values
        bit_flip_mutation, # for binary values
        bit_flip_mutation # for binary values
    ]
    algorithm = CustomDoeNSGA2(rng)

    if display and problem.objectives == 2:
        algorithm.observer = [initial_pop_observer_gray]
    else :
        algorithm.observer = initial_pop_observer_gray

    final_pop, final_pop_fitnesses = run_nsga2(rng, problem, variator, algorithm, display=display, num_vars=num_vars, **args)
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
    df.to_csv('final_pop_gray_doe.csv', index=False)
    
    ioff()
    show()
