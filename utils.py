from typing import List
import numpy as np
from bumblebee_pollination_abm.Model import GreenArea
from bumblebee_pollination_abm.Utils import PlantType, BeeType, BeeStage
from math import tanh

def getModel(no_mow_pc, mowing_days, pesticide_days, flower_area_type, seed):
    size = (50, 50)

    model_params = {
        "width": size[0], 
        "height": size[1], 
        "queens_quantity": 2, 
        "no_mow_pc": no_mow_pc,
        "steps_per_day": 40,
        "mowing_days": mowing_days,
        "pesticide_days": pesticide_days,
        "false_year_duration": 190, #false duration of the year withouth hibernation period
        "seed_max_age": {
            PlantType.SPRING_TYPE1: 0,
            PlantType.SPRING_TYPE2: 0,
            PlantType.SPRING_TYPE3: 0,
            PlantType.SUMMER_TYPE1: 70,
            PlantType.SUMMER_TYPE2: 70,
            PlantType.SUMMER_TYPE3: 70,
            PlantType.AUTUMN_TYPE1: 150,
            PlantType.AUTUMN_TYPE2: 150,
            PlantType.AUTUMN_TYPE3: 150
        },
        "plant_reward": {
            PlantType.SPRING_TYPE1: (0.45, 0.5),
            PlantType.SPRING_TYPE2: (0.35, 0.6),
            PlantType.SPRING_TYPE3: (0.4, 0.55),
            PlantType.SUMMER_TYPE1: (0.45, 0.5),
            PlantType.SUMMER_TYPE2: (0.35, 0.6),
            PlantType.SUMMER_TYPE3: (0.4, 0.55),
            PlantType.AUTUMN_TYPE1: (0.45, 0.5),
            PlantType.AUTUMN_TYPE2: (0.35, 0.6),
            PlantType.AUTUMN_TYPE3: (0.4, 0.55)
        },
        "woods_drawing": False,
        "data_collection": False,
        "flower_area_type": flower_area_type,
        "bumblebee_params": {
            "max_memory": 10,
            "days_till_sampling_mode": 3,
            "steps_colony_return": 10,
            "bee_age_experience": 10,
            "max_pollen_load": 20,
            "male_percentage": 0.3,
            "new_queens_percentage": 0.3,
            "nest_bees_percentage": 0.3,
            "max_egg": 12,
            "days_per_eggs": 5,
            "queen_male_production_period": 120,
            "hibernation_resources": (19, 19),
            "stage_days": {
                BeeStage.EGG: 4,
                BeeStage.LARVAE: 13, 
                BeeStage.PUPA: 13,
                BeeStage.BEE: {
                    BeeType.WORKER: 25,
                    BeeType.NEST_BEE: 30,
                    BeeType.MALE: 10,
                    BeeType.QUEEN: 20
                },
                BeeStage.QUEEN: 130
            },
            "steps_for_consfused_flower_visit": 3,
            "max_collection_ratio": 1,
            "hibernation_survival_probability": 0.5
        },
        "plant_params": {
            "nectar_storage": 100, 
            "pollen_storage": 100,
            "nectar_step_recharge": 0.015, #amount of recharge after a step
            "pollen_step_recharge": 0.015, #amount of recharge after a step
            "flower_age": {
                PlantType.SPRING_TYPE1: 70,
                PlantType.SPRING_TYPE2: 70,
                PlantType.SPRING_TYPE3: 70,
                PlantType.SUMMER_TYPE1: 80,
                PlantType.SUMMER_TYPE2: 80,
                PlantType.SUMMER_TYPE3: 80,
                PlantType.AUTUMN_TYPE1: 40, # it's important that the sum coincides with false year duration
                PlantType.AUTUMN_TYPE2: 40,
                PlantType.AUTUMN_TYPE3: 40
            },
            "initial_seed_prod_prob": 0.2, #initial probability of seed production (it takes into account the wind and rain pollination)
            "max_seeds": 6, #maximum number of seeds produced by the flower
            "seed_prob": 0.6, #probability of a seed to become a flower
            "max_gen_per_season": 2
        },
        "colony_params": {
            "nectar_consumption_per_bee": 0.7,
            "pollen_consumption_per_bee": 0.7,
            "days_till_death": 4
        },
        "seed": seed
    }

    return GreenArea(**model_params)

def runModel(args, generation):
    '''
    args:
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
    '''

    model = getModel(**args)
    # 7600 steps per year
    for i in range(22800): # 3 years
        model.step()

    if model.data_collection:
        saveData(model, generation)

    return model.getHibernatedQueensQuantity()

def saveData(model, generation):
    col_ag_df = model.datacollector_colonies.get_agent_vars_dataframe()
    bumb_mod_df = model.datacollector_bumblebees.get_model_vars_dataframe()
    pla_mod_df = model.datacollector_plants.get_model_vars_dataframe()
    col_ag_df.to_csv(
        f'col_ag_df_gen_{generation}.zip', 
        index=False, 
        compression={"method":'zip', "archive_name":f'col_ag_df_gen_{generation}.csv'}
    )
    bumb_mod_df.to_csv(
        f'bumb_mod_df_gen_{generation}.zip',
        index=False,
        compression={"method":'zip', "archive_name":f'bumb_mod_df_gen_{generation}.csv'}
    )
    pla_mod_df.to_csv(
        f'pla_mod_df_gen_{generation}.zip',
        index=False,
        compression={"method":'zip', "archive_name":f'pla_mod_df_gen_{generation}.csv'}
    )

def grayToDecimal(n: List[int]) -> int:
	n = ''.join(str(i) for i in n)
	n = int(n, 2)
	binary = 0

	# Taking xor until n becomes zero
	while (n):
		binary = binary ^ n
		n = n >> 1

	return binary

def grayCode(n: int, bits: int) -> List[int]:
    gray = n ^ (n >> 1)
    gray = bin(gray)[2:]
    gray_list = list(map(int, gray))
    gray_list = [0] * (bits - len(gray_list)) + gray_list
    return gray_list

def lhs_generator(random, args):
    lower_bounds = args["min"]
    upper_bounds = args["max"]
    points = []
    n = args["dim"]
    m = args["size"]
    for i in range(n):
        interval = (upper_bounds[i]-lower_bounds[i])/m
        lower_limits = np.arange(start=lower_bounds[i], stop=upper_bounds[i], step=interval)
        upper_limits = np.arange(start=(lower_bounds[i]+interval), stop=(upper_bounds[i]+interval), step=interval)
        p = random.uniform(low = lower_limits, high = upper_limits, size = m)
        random.shuffle(p)
        points.append(list(p))
    if(args["dim"]==1):
      points = points[0]
    return np.array(points)

def getLiveability(x1,x2,x3,x4):
    flower_area_type_points = { # 0 is the best, 1 is the worst
        0: 0.1, #centered square
        1: 0.1, #centered circle
        2: 0.4, #north and south
        3: 0.2, #north
        4: 0.2, #south
        5: 0.2, #west
        6: 0.2 #east
    }
    alpha1 = 0.8
    alpha2 = 0.2
    beta1 = 1.5
    beta2 = 25
    kappa1 = 0.75
    kappa2 = 0.25
    s1 = x1
    s2 = flower_area_type_points[x4]
    g = 1 if (s1 ==1 or s2 == 1) else kappa1*s1 + kappa2*s2
    t2 = 60/(1+g)
    t3 = 60
    m2 = 189
    m3 = 189
    gamma1 = 0.1 * (1 - g) * (m2/(t2*g)) if g != 0 else 0
    gamma2 = 5
    w = (beta1*x3)/m3 if x3 <= t3 else (beta1*t3)/m3 + tanh((x3-t3)*beta2/m3)*((m3 - beta1*t3)/m3)
    e = (gamma1*g*x2)/m2 + g if x2 <= t2 else g + (gamma1*g*t2)/m2 + tanh((x2-t2)*gamma2/m2)*(((m2 - gamma1*g*t2)/m2) - g)
    f2 = 1 - (alpha1*e + alpha2*w)
    return f2