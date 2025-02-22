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


if __name__ == '__main__':
    x = np.linspace(-2, 4, 45)
    #print(x)
    y = f(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro')
    plt.show()