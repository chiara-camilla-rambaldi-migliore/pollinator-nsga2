import inspyred.ec as ec
from custom_evolve import CustomEvolutionaryComputation
class CustomNSGA2(ec.emo.NSGA2):
    """Evolutionary computation representing the nondominated sorting genetic algorithm.
    
    This class represents the nondominated sorting genetic algorithm (NSGA-II)
    of Kalyanmoy Deb et al. It uses nondominated sorting with crowding for 
    replacement, binary tournament selection to produce *population size*
    children, and a Pareto archival strategy. The remaining operators take 
    on the typical default values but they may be specified by the designer.
    
    """
    def __init__(self, random):
        super().__init__(random)
    
    def evolve(self, generator, evaluator, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        args.setdefault('num_selected', pop_size)
        args.setdefault('tournament_size', 2)
        return CustomEvolutionaryComputation.evolve(self, generator, evaluator, pop_size, seeds, maximize, bounder, **args)