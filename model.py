import numpy as np
from activations import get_activation_function, get_activation_derivative, softmax

class FeedforwardNN:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.activation = get_activation_function(activation)
        self.weights = {}
        self.biases = {}
        
        # Create layer sizes: input, hidden layers, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            if weight_init == 'random':
                self.weights[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01
            elif weight_init == 'xavier':
                self.weights[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / (layer_sizes[i-1] + layer_sizes[i]))
            self.biases[f'b{i}'] = np.zeros((1, layer_sizes[i]))
    
    def forward(self, X):
        """Forward propagation"""
        cache = {}
        cache['A0'] = X
        A = X
        L = len(self.hidden_sizes) + 1
        # Forward pass through hidden layers
        for i in range(1, L):
            Z = np.dot(A, self.weights[f'W{i}']) + self.biases[f'b{i}']
            cache[f'Z{i}'] = Z
            A = self.activation(Z)
            cache[f'A{i}'] = A
        # Output layer with softmax activation
        ZL = np.dot(A, self.weights[f'W{L}']) + self.biases[f'b{L}']
        cache[f'Z{L}'] = ZL
        A_L = softmax(ZL)
        cache[f'A{L}'] = A_L
        self.cache = cache
        return A_L

    def backward(self, Y):
        """Backward propagation. Compute gradients for weights and biases."""
        grads = {}
        L = len(self.hidden_sizes) + 1
        m = Y.shape[0]
        A_L = self.cache[f'A{L}']
        dZ = A_L - Y  
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f'A{i-1}']
            grads[f'dW{i}'] = np.dot(A_prev.T, dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 1:
                W = self.weights[f'W{i}']
                dA_prev = np.dot(dZ, W.T)
                Z_prev = self.cache[f'Z{i-1}']
                dZ = dA_prev * get_activation_derivative(self.activation_name, Z_prev)
        return grads

    def update_parameters(self, grads, optimizer):
        """
        Updates parameters using the optimizer provided.
        """
        num_layers = len(self.hidden_sizes) + 1
        for i in range(1, num_layers + 1):
            self.weights[f'W{i}'] = optimizer.update_param(self.weights[f'W{i}'], grads[f'dW{i}'], f'W{i}')
            self.biases[f'b{i}'] = optimizer.update_param(self.biases[f'b{i}'], grads[f'db{i}'], f'b{i}')
