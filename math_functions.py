import matplotlib.pyplot as plt
import numpy as np

def sigmoid_fn(x):
    z = 1 / (1 + np.exp(-x))
    return z

def relu_fn(x):
    res = np.where(x <0, 0, x)
    return res

def leaky_relu_fn(x):
    alpha=0.02
    res = np.where(x < 0, alpha*x, x)
    return res

def logit_fn(x):
    res = np.log(x) - np.log(1-x)
    return res

def tanh_fn(x):
    sinh = np.exp(x) - np.exp(-x)
    cosh = np.exp(x) + np.exp(-x)
    res = sinh/cosh
    return res

def softmax_fn(x):
    exp_arr = np.exp(x)
    res = np.exp(x) / exp_arr.sum()
    return res


def _plotXY(X,y, title):
    plt.title(title)
    plt.plot(X, y)
    plt.show()


def main():
    X = np.linspace(-4,4, 100)
    #y= sigmoid_fn(X)
    #y = relu_fn(X)
    #y = logit_fn(X)
    #y = leaky_relu_fn(X)
    #y = tanh_fn(X)
    y = softmax_fn(X)
    _plotXY(X,y,"softmax")

if __name__== "__main__":
    main()