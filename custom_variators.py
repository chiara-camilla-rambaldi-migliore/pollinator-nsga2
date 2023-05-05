from inspyred.ec.variators.crossovers import crossover
from inspyred.ec.variators.mutators import mutator
import copy

@crossover
def single_point_crossover(random, mom, dad, args):
    """Return the offspring of single-point crossover on the candidates.

    This function performs single-point crossover used for binary candidates.
    If choose a crossover point at random and swap the bits after that point.

    .. Arguments:
       random -- the random number generator object
       mom -- the first parent candidate
       dad -- the second parent candidate
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *crossover_rate* -- the rate at which crossover is performed 
      (default 1.0)
    - *spx_points* -- a list of points specifying the alleles to
      recombine (default None)
    
    """
    spx_points = args.setdefault('spx_points', None)
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    bounder = args['_ec'].bounder
    children = []
    if random.random() < crossover_rate:
        bro = copy.copy(dad)
        sis = copy.copy(mom)
        if spx_points is None:
            spx_points = list(range(min(len(bro), len(sis))))
        index = random.choice(spx_points)
        for i in spx_points:
            if i >= index:
                bro[i], sis[i] = sis[i], bro[i]
        bro = bounder(bro, args)
        sis = bounder(sis, args)
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children

@mutator
def bit_flip_mutation(random, candidate, args):
    """Return the mutants produced by bit-flip mutation on the candidates.

    This function performs bit-flip mutation. If a candidate solution contains
    non-binary values, this function leaves it unchanged.

    .. Arguments:
       random -- the random number generator object
       candidate -- the candidate solution
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *mutation_rate* -- the rate at which mutation is performed (default 0.1)
    - *bf_points* -- a list of points specifying the alleles to
      mutate (default None)
    
    The mutation rate is applied on a bit by bit basis.
    
    """
    rate = args.setdefault('mutation_rate', 0.1)
    bf_points = args.setdefault('bf_points', None)
    mutant = copy.copy(candidate)
    bounder = args['_ec'].bounder
    if bf_points is None:
      bf_points = [i for i in range(len(mutant)) if mutant[i] in [0, 1]]
    for i in bf_points:
      if random.random() < rate:
        mutant[i] = (mutant[i] + 1) % 2
    
    mutant = bounder(mutant, args)
    return mutant

@mutator    
def gaussian_mutation(random, candidate, args):
    """Return the mutants created by Gaussian mutation on the candidates.

    This function performs Gaussian mutation. This function  
    makes use of the bounder function as specified in the EC's 
    ``evolve`` method.

    .. Arguments:
       random -- the random number generator object
       candidate -- the candidate solution
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *mutation_rate* -- the rate at which mutation is performed (default 0.1)
    - *gaussian_mean* -- the mean used in the Gaussian function (default 0)
    - *gaussian_stdev* -- the standard deviation used in the Gaussian function
      (default 1)
    - *g_points* -- a list of points specifying the alleles to
      mutate (default None)
      
    The mutation rate is applied on an element by element basis.
    
    """
    mut_rate = args.setdefault('mutation_rate', 0.1)
    mean = args.setdefault('gaussian_mean', 0.0)
    stdev = args.setdefault('gaussian_stdev', 1.0)
    g_points = args.setdefault('g_points', None)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    if g_points is None:
        g_points = list(range(len(mutant)))
    for i in g_points:
        if random.random() < mut_rate:
            mutant[i] += random.gauss(mean, stdev)
    mutant = bounder(mutant, args)
    return mutant