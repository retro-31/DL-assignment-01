import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def identity(x):
    return x

def get_activation_function(name):
    if name == 'sigmoid':
        return sigmoid
    elif name == 'relu':
        return relu
    elif name == 'tanh':
        return tanh
    elif name == 'identity':
        return identity
    else:
        # default
        return sigmoid
