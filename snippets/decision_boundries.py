import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def plot_diagonal_db():
    x_0 = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 1, 0.8, 0.1]
    y_0 = [0.2, 0.4, 0.2, 0.7, 0.71, 0.9, 0.83, 0.1, 0.9]
    x_1 = [1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 1.3, 1.2, 1.1, 2, 1.9, 1.7]
    y_1 = [1.4, 1.52, 1.8, 1.3, 1.7, 1.95, 1.9, 1.8, 1.6, 1.9, 1.2, 1.1, 1.4]
    plt.scatter(x_0, y_0, label="Class 0")
    plt.scatter(x_1, y_1, label="Class 1")
    x0 = np.arange(0, 2, step=0.1)
    x1 = np.arange(2, 0, step=-0.1)
    plt.plot(x0, x1, label="Decision Boundry", color="r")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.savefig("./assets/diagonal.png")
    # plt.show()


def plot_circular_db():
    c_0 = list(product(
        np.arange(-0.8, 0.8, step=0.1), np.arange(-0.8, 0.8, step=0.1)))
    x_0 = [x*np.random.random(1)[0] for x,_ in c_0]
    x_1 = [y*np.random.random(1)[0] for _,y in c_0]
    f_ = lambda x: -2 + x * np.random.random(1)[0]+ x**2
    y_0 = list(np.arange(-2, 2, step=0.1))
    y_1 = [f_(x) for x in list(np.arange(-2, 2, step=0.1))]
    plt.scatter(x_0, x_1, label="Class 0")
    plt.scatter(y_0, y_1, label="Class 1")
    angle = np.linspace( 0 , 2 * np.pi , 150 )
    x = np.cos( angle )
    y = np.sin( angle )
    plt.plot(x, y, label="Decision Boundry", color="r")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.savefig("./assets/circular.png")
    # plt.show()



plot_diagonal_db()
plot_circular_db()
