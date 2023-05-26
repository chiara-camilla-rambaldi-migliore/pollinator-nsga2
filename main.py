import argparse
import os
from utils import grayToDecimal
from bumblebee_problem_gray import UrbanPollinator as UP_gray
from bumblebee_problem_gray_doe import UrbanPollinator as UP_gray_doe
from bumblebee_problem_value import UrbanPollinator as UP_value
from bumblebee_problem_value_doe import UrbanPollinator as UP_value_doe
from multi_objective_bumblebee import run_nsga2
from custom_variators import single_point_crossover, bit_flip_mutation, gaussian_mutation
from inspyred.ec import variators
from inspyred_utils import initial_pop_observer_gray, initial_pop_observer_value
from inspyred_utils import NumpyRandomWrapper
import logging
import pandas as pd
from custom_nsga2 import CustomNSGA2, CustomDoeNSGA2
import collections
collections.Iterable = collections.abc.Iterable
collections.Sequence = collections.abc.Sequence

parser = argparse.ArgumentParser(description='Plot fitnesses of same encoding but different seeds.')

parser.add_argument('-a', dest='algorithm', action='store',
                    help='Algorithm to run: "gray", "gray_doe", "value", "value_doe". Default is "gray"', default='gray')
parser.add_argument('-c', dest='max_cores', action='store',
                    help='Maximum number of cores to use. Default is 10', default=10, type=int)
parser.add_argument('-g', dest='generations', action='store', type=int,
                    help='Number of generations. Default is 30', default=30)
parser.add_argument('-n', dest='individuals_qty', action='store', type=int,
                    help='Quantity of individuals in population. Default is 10', default=10)
parser.add_argument('-s', dest='seed', action='store', type=int,
                    help='Seed to use. Default is 23', default=23)

args = parser.parse_args()


def execute_ga(args):
    ga_args = {
        "max_cores": args.max_cores,
        "seed": args.seed,
        "pop_size": args.individuals_qty,
        "max_generations": args.generations,
        "inspyred_log_filename": os.path.join("inspyred_logs", f'inspyred_{args.algorithm}_{args.individuals_qty}x{args.generations}_{args.seed}.log'),
        "initial_pop_filename": os.path.join("pops", f'pop_{args.algorithm}_{args.individuals_qty}x{args.generations}_{args.seed}.csv'),
        "final_pop_filename": os.path.join("final_pops", f'final_pop_{args.algorithm}_{args.individuals_qty}x{args.generations}_{args.seed}.csv'),
    }

    # make a directory if doesn't exist
    os.makedirs(os.path.dirname(ga_args["inspyred_log_filename"]), exist_ok=True)
    os.makedirs(os.path.dirname(ga_args["initial_pop_filename"]), exist_ok=True)
    os.makedirs(os.path.dirname(ga_args["final_pop_filename"]), exist_ok=True)

    execute_ga_function = globals()[f"get_{args.algorithm}_args"]
    ga_args = execute_ga_function(ga_args)
    execute(ga_args)

def get_generic_gray_args(ga_args):
    ga_args["variations_args"] = [
        {"blx_points": [0]},
        {"spx_points": [1, 2, 3, 4, 5, 6, 7, 8]},
        {"spx_points": [9,10,11,12,13,14,15,16]},
        {"spx_points": [17,18,19]},
        {"g_points": [0], "gaussian_stdev": 0.1}, # 95% of gaussian values are in the range [-0.2, 0.2]
        {"bf_points": [1, 2, 3, 4, 5, 6, 7, 8]},
        {"bf_points": [9,10,11,12,13,14,15,16]},
        {"bf_points": [17,18,19]},
    ]
    ga_args["num_vars"] = 20
    ga_args["variator"] = [
        variators.blend_crossover, # for no mow percentual
        single_point_crossover, # for mowing days
        single_point_crossover, # for pesticide days
        single_point_crossover, # for flower area type
        gaussian_mutation, # for real values
        bit_flip_mutation, # for binary values
        bit_flip_mutation, # for binary values
        bit_flip_mutation # for binary values
    ]
    ga_args["observer"] = initial_pop_observer_gray
    ga_args["final_pop_function"] = get_final_pop_gray
    return ga_args

def get_generic_value_args(ga_args):
    ga_args["variations_args"] = [
        {"blx_points": [0, 1, 2, 3]},
        {"g_points": [0], "gaussian_stdev": 0.1}, #95% of gaussian values are in the range [-0.2, 0.2]
        {"g_points": [1, 2, 3], "gaussian_stdev": 10}, #95% of gaussian values are in the range [-20, 20]
    ]
    ga_args["num_vars"] = 4
    ga_args["variator"] = [
        variators.blend_crossover, # for no mow percentual
        gaussian_mutation, # for real values
        gaussian_mutation, # for integer values
    ]
    ga_args["observer"] = initial_pop_observer_value
    ga_args["final_pop_function"] = get_final_pop_value
    return ga_args

def get_gray_args(ga_args):
    ga_args = get_generic_gray_args(ga_args)
    ga_args["problem"] = UP_gray
    ga_args["algorithm"] = CustomNSGA2
    return ga_args

def get_gray_doe_args(ga_args):
    ga_args = get_generic_gray_args(ga_args)
    ga_args["problem"] = UP_gray_doe
    ga_args["algorithm"] = CustomDoeNSGA2
    return ga_args

def get_value_args(ga_args):
    ga_args = get_generic_value_args(ga_args)
    ga_args["problem"] = UP_value
    ga_args["algorithm"] = CustomNSGA2
    return ga_args

def get_value_doe_args(ga_args):
    ga_args = get_generic_value_args(ga_args)
    ga_args["problem"] = UP_value_doe
    ga_args["algorithm"] = CustomDoeNSGA2
    return ga_args

def get_final_pop_gray(final_pop, final_pop_fitnesses):
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
    return new_final_pop

def get_final_pop_value(final_pop, final_pop_fitnesses):
    new_final_pop = []
    for i, guy in enumerate(final_pop):
        new_guy = guy
        new_guy.append(final_pop_fitnesses[i][0])
        new_guy.append(final_pop_fitnesses[i][1])
        new_final_pop.append(new_guy)
    return new_final_pop
    

def execute(ga_args):
    # parameters for NSGA-2
    args = {}
    args["pop_size"] = ga_args["pop_size"]
    args["max_generations"] = ga_args["max_generations"]

    args["variations_args"] = ga_args["variations_args"]

    args["max_cores"] = ga_args["max_cores"]
    args["seed"] = ga_args["seed"]

    rng = NumpyRandomWrapper(args["seed"])

    args["fileName_initial_pop"] = ga_args["initial_pop_filename"]
    
    logger = logging.getLogger('inspyred.ec')
    logger.setLevel(logging.DEBUG)
    log_filename = ga_args["inspyred_log_filename"]
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    algorithm = ga_args["algorithm"](rng)

    algorithm.observer = ga_args["observer"]

    final_pop, final_pop_fitnesses = run_nsga2(ga_args["problem"](), ga_args["variator"], algorithm, num_vars=ga_args["num_vars"], **args)
    new_final_pop = ga_args["final_pop_function"](final_pop, final_pop_fitnesses)
    
    df = pd.DataFrame(new_final_pop, columns=['no_mow_pc', 'mowing_days', 'pesticide_days', 'flower_area_type', 'fitness_1', 'fitness_2'])
    final_pop_filename = ga_args["final_pop_filename"]
    df.to_csv(final_pop_filename, index=False)


if __name__ == "__main__":
    execute_ga(args)