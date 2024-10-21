import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# parabola function
def parabola_fn(x):
    return x ** 2

def _derivative(func, x, h=1e-5):
    # h is infinetly small. dx
    # dydx = 1/h * (func(x + h) - func(x))
    return (func(x+h) - func(x)) * 1/h


# visualize a gradient descent with a specified learning rate
def gradient_descent(X, Y, lr=0.001):
    # randomly select a starting point
    start_xy_pos = (90, parabola_fn(90))
    for i in range(1000):
        index = np.where(X==start_xy_pos[0])[0]
        dydx = _derivative(parabola_fn,X[index-i])
        if dydx <= 0: # reached mimina
            time.sleep(10) # pause the plot
            break
        curr_x = X[index-i] - (lr * _derivative(parabola_fn, X[index-i]))
        curr_y = parabola_fn(curr_x)
        plt.plot(X,Y)
        plt.scatter(curr_x, curr_y, c="red")
        plt.pause(0.0001)
        plt.clf()
      

def main():
    # create an array and plot the parabola
    x = np.arange(start=-100, stop=100)
    y = parabola_fn(x)
    gradient_descent(X=x, Y=y)
    #gradient_descent_animate(x,y)

if __name__ == "__main__":
    main()