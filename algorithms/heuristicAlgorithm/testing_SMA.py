# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:38:52 2024

@author: richie bao
"""
import numpy as np
# from abc import ABC, abstractmethod
import numbers as nb

from utils.space import FloatVar
from bio_based import SMA



def objective_function(solution):
    return np.sum(solution**2)

if __name__=="__main__":
    # bounds=FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta")
    #a=bounds.generator
    #print(a.random())
    
    problem_dict = {
        "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
        "minmax": "min",
        "obj_func": objective_function
        }
            

    model = SMA.DevSMA(epoch=1000, pop_size=50, p_t = 0.03)
    
    # print(model.__dict__)     
    
    g_best = model.solve(problem_dict)
    
    
    
    
    # print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    # print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")    

    

