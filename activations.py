import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
        return sigmoid

def get_activation_derivative(name, x):
    if name == 'sigmoid':
        return sigmoid_derivative(x)
    elif name == 'relu':
        return relu_derivative(x)
    elif name == 'tanh':
        return tanh_derivative(x)
    elif name == 'identity':
        return identity_derivative(x)
    else:
        return sigmoid_derivative(x)