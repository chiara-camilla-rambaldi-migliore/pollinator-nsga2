from pylab import asarray
from inspyred.ec import terminators


def run_nsga2(problem, variator, algorithm, num_vars=0, use_bounder=True, **kwargs) :
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

    return final_pop_candidates, final_pop_fitnesses

