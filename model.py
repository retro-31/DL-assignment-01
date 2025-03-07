import numpy as np
from activations import get_activation_function, softmax

class FeedforwardNN:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.activation = get_activation_function(activation)
        self.weights = {}
        self.biases = {}
        
        # Create list of layer sizes: input, hidden layers, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases for each layer
        for i in range(1, len(layer_sizes)):
            if weight_init == 'random':
                self.weights[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01
            elif weight_init == 'xavier':
                self.weights[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / (layer_sizes[i-1] + layer_sizes[i]))
            self.biases[f'b{i}'] = np.zeros((1, layer_sizes[i]))
            
    def forward(self, X):
        # input layer
        A = X

        # hidden layers
        L = len(self.hidden_sizes) + 1
        for i in range(1, L):
            Z = np.dot(A, self.weights[f'W{i}']) + self.biases[f'b{i}']
            A = self.activation(Z)

        # output layer
        ZL = np.dot(A, self.weights[f'W{L}']) + self.biases[f'b{L}']
        out = softmax(ZL)

        return out
