'''

    Explore the distribution of wind speed over a couple days.

'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
import num2words

mpl.style.use('seaborn')


# Pull the wind data.
data = __import__('dataset').wind_dist_data()

for i in range(1, 4):
    wind_diffs = data[:-i, 3] - data[i:, 3]

    print(f'Mean: {wind_diffs.mean()}, variance: {wind_diffs.std() ** 2}')

    # Min-Max normalization in range [-1, 1].
    wind_diffs = -1 + ((wind_diffs - min(wind_diffs)) * 2) / (max(wind_diffs) - min(wind_diffs))

    print(f'Normalized mean: {wind_diffs.mean()}, variance: {wind_diffs.std() ** 2}')


    plt.subplot(1, 3, i)
    plt.hist(wind_diffs, np.linspace(min(wind_diffs), max(wind_diffs), 30))
    plt.title(f'{num2words.num2words(i)} day(s)')

plt.show()



