from inspyred.benchmarks import Benchmark
from utils import runModel
from custom_bounder import Bounder
from inspyred.ec import emo
import concurrent.futures as futures
from utils import getLiveability, lhs_generator

class UrbanPollinator(Benchmark):
    """
        Defines the Urban Pollinator multiobjective problem.

        4 dimensions: 
        "no_mow_pc": percentage of area not mowed, it is a value between 0 and 1;
        "mowing_days": number of days between mowing, it is a value between 0 and 190;
        "pesticide_days": number of days between pesticide application, it is a value between 0 and 190;
        "flower_area_type": type of flower area 
            (
                1: flower area in the center in the form of square, 
                2: flower area in the center in the form of a circle, 
                3: flower area in the north and south section, 
                4: flower area in the north section, 
                5: flower area in the south section, 
                6: flower area in the west section, 
                7: flower area in the east section
            )
    """
    def __init__(self):
        Benchmark.__init__(self, 4, 2)
        self.bounder = Bounder(
            [0, 1, 2, 3], 
            [[0,1], [1,190], [1,190], [1,7]], 
            ['value_coded', 'value_coded_discrete', 'value_coded_discrete', 'value_coded_discrete']
        )
        self.maximize = True

    def generator(self, random, qty, args):
        gen_args = {'size': qty, 'dim': self.dimensions, 'min' : [0,1,1,1], 'max': [1, 190, 190, 7]}
        seq = lhs_generator(random, gen_args)
        seq = [
            [
                seq[0][i],
                round(seq[1][i]),
                round(seq[2][i]),
                round(seq[3][i])
            ]
            for i in range(qty)
        ]
        return seq
        
    def evaluator(self, candidates, args):
        generation = args.setdefault('generation', 0)
        seed = args.setdefault('seed', 23)
        max_cores = args.setdefault('max_cores', 8)
        fitness = []
        f1s = self.getFitness1(candidates, generation, seed, max_cores)
        f2s = self.getFitness2(candidates)
        for i in range(len(candidates)):
            fitness.append(emo.Pareto([f1s[i], f2s[i]]))

        return fitness
    
    def getFitness1(self, candidates, generation, seed, max_cores=8):
        fitness = []

        # Multiprocessing
        proc_res = []

        with futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            for c in candidates:
                args = {
                    "no_mow_pc": c[0],
                    "mowing_days": c[1],
                    "pesticide_days": c[2],
                    "flower_area_type": c[3],
                    "seed": seed
                }
                proc_res.append(
                    executor.submit(
                        runModel, args, generation
                    )
                )

        for i in range(len(proc_res)):
            try:
                qty = proc_res[i].result()
                fitness.append(qty)
                print(f"Process {i} terminated correctly")
            except Exception as ex:
                print(f"Error in process {i}: [{ex}]")

            
        return fitness
    
    def getFitness2(self, candidates):
        fitness = []
        for c in candidates:
            x1 = c[0] #no_mow_pc
            x2 = c[1] - 1 #mowing_days
            x3 = c[2] - 1 #pesticide_days
            x4 = c[3] - 1 #flower_area_type
            
            f2 = getLiveability(x1, x2, x3, x4)

            fitness.append(f2)
        return fitness