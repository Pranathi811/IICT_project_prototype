import itertools
import numpy as np
from src.predict import predict_yield

def recommend_conditions(temp, pressure, reaction_time,
                         concentration, catalyst):

    best_yield = -1
    best_input = None

    temp_range = np.arange(temp-20, temp+21, 10)
    pressure_range = np.arange(pressure-5, pressure+11, 2)
    time_range = np.arange(max(1, reaction_time-1), reaction_time+3, 1)
    conc_range = np.arange(max(0.1, concentration-0.5), concentration+1.0, 0.2)
    catalyst_range = np.arange(max(0.1, catalyst-0.2), catalyst+1.0, 0.2)

    for values in itertools.product(
        temp_range,
        pressure_range,
        time_range,
        conc_range,
        catalyst_range
    ):

        x = np.array([values])

        pred = predict_yield(x)

        if pred > best_yield:
            best_yield = pred
            best_input = values

    return best_input, best_yield