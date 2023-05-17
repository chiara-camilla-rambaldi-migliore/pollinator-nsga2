import inspyred.ec as ec
from custom_evolve import CustomEvolutionaryComputation
from custom_evolve_doe import CustomDoeEvolutionaryComputation

class CustomNSGA2(ec.emo.NSGA2):
    def __init__(self, random):
        super().__init__(random)
    
    def evolve(self, generator, evaluator, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        args.setdefault('num_selected', pop_size)
        args.setdefault('tournament_size', 2)
        return CustomEvolutionaryComputation.evolve(self, generator, evaluator, pop_size, seeds, maximize, bounder, **args)
    
class CustomDoeNSGA2(ec.emo.NSGA2):
    def __init__(self, random):
        super().__init__(random)
    
    def evolve(self, generator, evaluator, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        args.setdefault('num_selected', pop_size)
        args.setdefault('tournament_size', 2)
        return CustomDoeEvolutionaryComputation.evolve(self, generator, evaluator, pop_size, seeds, maximize, bounder, **args)