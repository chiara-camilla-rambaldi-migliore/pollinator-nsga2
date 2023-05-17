from pylab import *

from inspyred.ec.emo import Pareto
from numpy.random import RandomState
import pandas as pd
from utils import grayToDecimal

import functools

def choice_without_replacement(rng, n, size) :
    result = set()
    while len(result) < size :
        result.add(rng.randint(0, n))
    return result

class NumpyRandomWrapper(RandomState):
    def __init__(self, seed=None):
        super(NumpyRandomWrapper, self).__init__(seed)
        
    def sample(self, population, k):
        if isinstance(population, int) :
            population = range(population)           
        
        return asarray([population[i] for i in 
                        choice_without_replacement(self, len(population), k)])
        #return #self.choice(population, k, replace=False)
        
    def random(self):
        return self.random_sample()
    
    def gauss(self, mu, sigma):
        return self.normal(mu, sigma)
    
def initial_pop_observer_gray(population, num_generations, num_evaluations, 
                         args):
    for guy in population:
        new_guy = []
        new_guy.append(round(guy.candidate[0], 3))
        new_guy.append(grayToDecimal(guy.candidate[1:9]))
        new_guy.append(grayToDecimal(guy.candidate[9:17]))
        new_guy.append(grayToDecimal(guy.candidate[17:]))
        new_guy.append(guy.fitness[0])
        new_guy.append(round(guy.fitness[1], 3))
        args["initial_pop_storage"].append(new_guy)

    df = pd.DataFrame(args["initial_pop_storage"], columns=['no_mow_pc', 'mowing_days', 'pesticide_days', 'flower_area_type', 'fitness_1', 'fitness_2'])
    df.to_csv(args["fileName_initial_pop"], index=False)

def initial_pop_observer_value(population, num_generations, num_evaluations, 
                         args):
    for guy in population:
        new_guy = []
        new_guy.append(round(guy.candidate[0], 3))
        new_guy.append(guy.candidate[1])
        new_guy.append(guy.candidate[2])
        new_guy.append(guy.candidate[3])
        new_guy.append(guy.fitness[0])
        new_guy.append(round(guy.fitness[1], 3))
        args["initial_pop_storage"].append(new_guy)

    df = pd.DataFrame(args["initial_pop_storage"], columns=['no_mow_pc', 'mowing_days', 'pesticide_days', 'flower_area_type', 'fitness_1', 'fitness_2'])
    df.to_csv(args["fileName_initial_pop"], index=False)
    
        
def generator(random, args):
    return asarray([random.uniform(args["pop_init_range"][0],
                                   args["pop_init_range"][1]) 
                    for _ in range(args["num_vars"])])

def generator_wrapper(func):
        @functools.wraps(func)
        def _generator(random, args):
            return asarray(func(random, args))
        return _generator

class CombinedObjectives(Pareto):
    def __init__(self, pareto, args):
        """ edit this function to change the way that multiple objectives
        are combined into a single objective
        
        """
        
        Pareto.__init__(self, pareto.values)
        if "fitness_weights" in args :
            weights = asarray(args["fitness_weights"])
        else : 
            weights = asarray([1 for _ in pareto.values])
        
        self.fitness = sum(asarray(pareto.values) * weights)
        
    def __lt__(self, other):
        return self.fitness < other.fitness
        
def single_objective_evaluator(candidates, args):
    problem = args["problem"]
    return [CombinedObjectives(fit,args) for fit in 
            problem.evaluator(candidates, args)]
