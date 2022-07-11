import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def step_function(x):
    if x<0:
        return 0
    else:
        return 1

training_set = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]