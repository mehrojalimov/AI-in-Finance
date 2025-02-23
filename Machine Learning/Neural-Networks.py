import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

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

if __name__ == '__main__':
    x = np.linspace(-2, 4, 45)
    #print(x)
    y = f(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label= 'sample data')

    plt.plot(x, simple_linear_regresion(x, y), lw=3.0, label='linear regression')
    plt.legend()


    for deg in [1, 2, 3]:
        reg = np.polyfit(x, y, deg)
        y_ = np.polyval(reg, x)
        MSE_ = MSE(y, y_)
        print(f' deg = {deg} | MSE = {MSE_:.5f}')
        plt.plot(x, y_, label=f'deg = {deg}')
    #deg = 1 | MSE = 9.96182
    #deg = 2 | MSE = 2.10406
    #deg = 3 | MSE = 0.00000

    print(reg)
    #This is the optimal ("perfect") paramether values
    #[-0.3333  2.      0.     -0.    ]
    plt.show()