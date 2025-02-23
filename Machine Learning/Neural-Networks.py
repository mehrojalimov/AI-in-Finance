import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neural_network import MLPRegressor

sns.set_theme()
plt.style.use('seaborn-v0_8-darkgrid')

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

np.set_printoptions(precision=4, suppress=True)

def f(x):
    return 2 * x ** 2 - x ** 3 /3

def simple_linear_regresion(x, y):
    betta = np.cov(x, y)[0,1] / np.var(x)
    alpha = np.mean(y) - betta * np.mean(x)
    y_ = alpha + betta * x
    return y_

def MSE(y, y_):
    return np.mean((y - y_) ** 2)

def poly_regresion(x, y, deg):
    #The np.polyfit function from the NumPy library is used to fit a polynomial of a specified degree to a set of data points.
    #It returns the coefficients of the polynomial that best fits the data in a least-squares sense.
    reg = np.polyfit(x, y, deg)
    #print(reg)
    #Here, np.polyval takes the coefficients reg obtained from np.polyfit and evaluates the polynomial at the x-coordinates x, 
    #returning the predicted y-coordinates y_
    y_ = np.polyval(reg, x)
    return y_

def neural_network(x, y):
    model = MLPRegressor(hidden_layer_sizes=3 * [256], learning_rate_init=0.03, max_iter=5000)
    model.fit(x.reshape(-1, 1), y)
    MLPRegressor(hidden_layer_sizes=[256, 256, 256], learning_rate_init=0.03, max_iter=5000)
    y_ = model.predict(x.reshape(-1, 1))
    return y_

def draw(x, y, y_, label):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='sample data')
    plt.plot(x, y_, lw=3.0, label=label)
    plt.legend()
    plt.show()

def draw_poly(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='sample data')
    for deg in [1, 2, 3]:
        y_ = poly_regresion(x, y, deg)
        #print(y_)
        #print(y)
        MSE_ = MSE(y, y_)
        print(f' deg = {deg} | MSE = {MSE_:.5f}')
        plt.plot(x, y_, label=f'deg = {deg}')
    plt.legend()
    plt.show()


def draw_neural_network(x, y):
    y_ = neural_network(x, y)
    plt.figure(figsize=(10, 6))

    MSE = ((y - y_) ** 2).mean()
    print(f'MSE = {MSE:.5f}')

    plt.plot(x, y, 'ro', label='sample data')
    plt.plot(x, y_, lw=3.0, label='neural network, dnn estimation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x = np.linspace(-2, 4, 45)
    y = f(x)
    
    #draw(x, y, simple_linear_regresion(x, y), 'linear regression')

    #draw_poly(x, y)
    #deg = 1 | MSE = 9.96182
    #deg = 2 | MSE = 2.10406
    #deg = 3 | MSE = 0.00000
    #This is the optimal ("perfect") paramether values
    #[-0.3333  2.      0.     -0.    ]

    draw_neural_network(x, y)
