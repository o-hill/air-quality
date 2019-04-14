'''

    Deterministic Model for air quality index.

'''

import numpy as np
from scipy import integrate

from dataset import build_df, get_dfs



# Only use 2017 for now.
df = build_df(*get_dfs(2017))




def simulate_year() -> None:
    '''Simulate a year of changing AQI.'''

    aqi0 = df.loc[0]['CO AQI']
    t0, dt, tfinal = 0, 1, 250

    br = integrate.ode(model)
    br.set_initial_value(aqi0, t0)

    while br.successful and br.t < tfinal:
        br.integrate(br.t + dt)

    from ipdb import set_trace as debug; debug()




def model(t: int, x: tuple) -> np.ndarray:
    '''Calculate the RHS of the AQI ODE.'''

    aqi = x[0]

    dAQIdt = np.array([
        -0.01 * df.loc[int(np.floor(t))]['Wind Speed'] * aqi
    ])

    return dAQIdt


if __name__ == '__main__':
    simulate_year()
