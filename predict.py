'''

    Predict the next couple days of AQI, and plot it.

'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.style.use('seaborn')

from dataset import build_network_data, build_df

MEAN = 0
VAR = [2.56, 3.51, 3.82]
SVAR = [8.99, 13.09, 14.13]


def predict() -> None:

    # Get the data.
    X, Xtest, y, ytest = build_network_data()

    # Choose a random day we have data for.
    day = np.random.choice(ytest.shape[0] - 2)

    # Try a couple different methods of prediction.
    lspred = predict_least_squares(X, Xtest, y, ytest, day)


def predict_least_squares(X, Xtest, y, ytest, day) -> np.ndarray:

    regress = LinearRegression().fit(X, y)
    aqi = Xtest[day][2]
    wind_speed = Xtest[day][3]
    variates = Xtest[day]
    aqis = [ ]

    # Predict the next few days iteratively.
    for _ in range(1000):
        aqis.append([aqi])

        for i in range(3):
            aqis[-1].append(np.clip(regress.predict([variates]), 0, None))
            variates[3] += np.random.normal(MEAN, np.sqrt(VAR[i]))
            variates[1] += np.random.normal(MEAN, np.sqrt(SVAR[i]))
            variates[2] = aqis[-1][-1]

    plot_prediction(ytest[day - 1 : day + 3], np.mean(np.array(aqis), axis=0))



def plot_prediction(y_true: np.ndarray, y_hat: np.ndarray) -> None:

    plt.plot(range(y_true.shape[0]), y_true, label='Ground Truth')
    plt.plot(range(y_hat.shape[0]), y_hat, label='Prediction')

    plt.legend(loc='best')
    plt.title('True AQI vs. Predicted AQI for Three Days')
    plt.xlabel('Time (days)')
    plt.ylabel('Air Quality Index')
    plt.show()





if __name__ == '__main__':
    predict()








