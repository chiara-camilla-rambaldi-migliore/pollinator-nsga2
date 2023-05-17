from pylab import *

from inspyred.ec.emo import NSGA2
from inspyred.ec import terminators, variators, replacers, selectors
from custom_variators import single_point_crossover, bit_flip_mutation, gaussian_mutation
from custom_nsga2 import CustomNSGA2

import inspyred_utils
import plot_utils

def run_nsga2(random, problem, variator, algorithm, display=False, num_vars=0, use_bounder=True, **kwargs) :
    """ run NSGA2 on the given problem """
    
    #create dictionaries to store data about initial population, and lines
    initial_pop_storage = []
 
    algorithm.terminator = terminators.generation_termination 
    algorithm.variator = variator
    
    kwargs["num_selected"]=kwargs["pop_size"]  
    if use_bounder :
        kwargs["bounder"]=problem.bounder
        
    final_pop = algorithm.evolve(evaluator=problem.evaluator,  
                          maximize=problem.maximize,
                          initial_pop_storage=initial_pop_storage,
                          num_vars=num_vars, 
                          generator=problem.generator,
                          **kwargs) #kwargs will take also args for variators      
    
    final_pop_fitnesses = asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = [guy.candidate[0:num_vars] for guy in final_pop]

    if display :
        # Plot the parent and the offspring on the fitness landscape 
        # (only for 1D or 2D functions)
        if num_vars == 1 :
            plot_utils.plot_results_multi_objective_1D(problem, 
                                  initial_pop_storage["individuals"], 
                                  initial_pop_storage["fitnesses"], 
                                  final_pop_candidates, final_pop_fitnesses,
                                  'Initial Population', 'Final Population',
                                  len(final_pop_fitnesses[0]), kwargs)
    
        elif num_vars == 2 :
            plot_utils.plot_results_multi_objective_2D(problem, 
                                  initial_pop_storage["individuals"], 
                                  final_pop_candidates, 'Initial Population',
                                  'Final Population',
                                  len(final_pop_fitnesses[0]), kwargs)

        plot_utils.plot_results_multi_objective_PF(final_pop, kwargs['fig_title'] + ' (Pareto front)')

    return final_pop_candidates, final_pop_fitnesses

