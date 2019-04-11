'''

    Networks.

'''

import numpy as np

from keras.models import Model
from keras.layers import (
    Input,
    Dense
)

from dataset import build_network_data

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.style.use('seaborn')



def build_network(weights_path: str = ''):

    input_tensor = Input(shape=(7,))
    one = Dense(32)(input_tensor)
    two = Dense(64)(one)
    three = Dense(128)(two)
    four = Dense(128)(three)
    five = Dense(128)(four)
    six = Dense(64)(five)
    seven = Dense(32)(six)
    eight = Dense(8)(seven)
    output = Dense(1)(eight)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile('adadelta', loss='mean_squared_error')

    if weights_path:
        model.load_weights(weights_path)

    return model


def train_network() -> None:
    '''Train the network. Overwrites any previous weights.'''

    # Get the data and build the network from scratch.
    network = build_network()
    X, Xtest, y, y_test = build_network_data()

    converged = False
    it = 0
    while not converged:
        history = network.fit(x=X, y=y, batch_size=128, epochs=200)

        converged = history.history['loss'][0] < 0.000001
        network.save_weights('network_weights.h5')


def test_network() -> None:
    '''Test with the neural network.'''

    network = build_network('network_weights.h5')
    X, Xtest, y, y_test = build_network_data()
    y_predicted = network.predict(Xtest)

    results(y_test, y_predicted)



def results(y_test: np.ndarray, y_predicted: np.ndarray) -> None:
    '''Show the results of the predictions.'''

    print(f'Average distance between predictions and real values: {np.mean(np.abs(y_test - y_predicted))}')

    plt.plot(range(y_test.shape[0]), y_test, label='Real AQI')
    plt.plot(range(y_predicted.shape[0]), y_predicted, label='Predicted AQI')

    plt.xlabel('Index of Day in Year (2017)')
    plt.ylabel('Carbon Monoxide (CO) AQI')
    plt.title('Polynomial (degree 2) Regression Results for CO AQI in 2017')
    plt.legend(loc='best')
    plt.show()


def poly_regression():
    '''Try polynomial regression as well.'''

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    X, Xtest, y, ytest = build_network_data()

    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)
    Xtest_poly = poly.fit_transform(Xtest)

    regress = LinearRegression().fit(X_poly, y)
    y_predicted = np.clip(regress.predict(Xtest_poly), 0, None)

    print('Coefficients:')
    print(regress.coef_)
    results(ytest, y_predicted)



if __name__ == '__main__':
    # train_network()
    # test_network()

    poly_regression()



'''Notes

    Average distance for linear regression: 1.5479
    Average distance for degree 2 polynomial regression: 1.972
    Average distance for neural network: 3.8914

'''



















