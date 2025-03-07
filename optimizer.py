import numpy as np

class Optimizer:
    def __init__(self, optimizer_type='sgd', learning_rate=0.01, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum      # For momentum-based optimizers
        self.beta = beta              # For RMSprop
        self.beta1 = beta1            # For Adam/nadam (first moment)
        self.beta2 = beta2            # For Adam/nadam (second moment)
        self.epsilon = epsilon
        self.weight_decay = weight_decay  # L2 regularization parameter
        self.v = {}   # First moment estimates
        self.s = {}   # Second moment estimates
        self.t = {}   # Time steps for Adam/Nadam

    def update_param(self, param, grad, param_name):
        # weight decay regularization
        grad_reg = grad + self.weight_decay * param

        # Initialization of first and second moments with zeros
        if param_name not in self.v:
            self.v[param_name] = np.zeros_like(grad)
        if param_name not in self.s:
            self.s[param_name] = np.zeros_like(grad)
        if param_name not in self.t:
            self.t[param_name] = 0

        if self.optimizer_type == 'momentum':
            self.v[param_name] = self.momentum * self.v[param_name] - self.learning_rate * grad_reg
            param_updated = param + self.v[param_name]

        elif self.optimizer_type == 'nesterov':
            v_prev = self.v[param_name].copy()
            self.v[param_name] = self.momentum * self.v[param_name] - self.learning_rate * grad_reg
            param_updated = param - self.momentum * v_prev + (1 + self.momentum) * self.v[param_name]

        elif self.optimizer_type == 'rmsprop':
            self.s[param_name] = self.beta * self.s[param_name] + (1 - self.beta) * (grad_reg ** 2)
            param_updated = param - self.learning_rate * grad_reg / (np.sqrt(self.s[param_name]) + self.epsilon)

        elif self.optimizer_type == 'adam':
            self.t[param_name] += 1
            self.v[param_name] = self.beta1 * self.v[param_name] + (1 - self.beta1) * grad_reg
            self.s[param_name] = self.beta2 * self.s[param_name] + (1 - self.beta2) * (grad_reg ** 2)
            v_corrected = self.v[param_name] / (1 - self.beta1 ** self.t[param_name])
            s_corrected = self.s[param_name] / (1 - self.beta2 ** self.t[param_name])
            param_updated = param - self.learning_rate * v_corrected / (np.sqrt(s_corrected) + self.epsilon)

        elif self.optimizer_type == 'nadam':
            self.t[param_name] += 1
            self.v[param_name] = self.beta1 * self.v[param_name] + (1 - self.beta1) * grad_reg
            self.s[param_name] = self.beta2 * self.s[param_name] + (1 - self.beta2) * (grad_reg ** 2)
            v_corrected = self.v[param_name] / (1 - self.beta1 ** self.t[param_name])
            s_corrected = self.s[param_name] / (1 - self.beta2 ** self.t[param_name])
            param_updated = param - self.learning_rate * (self.beta1 * v_corrected + (1 - self.beta1) * grad_reg) / (np.sqrt(s_corrected) + self.epsilon)

        else:
            # Default SGD
            param_updated = param - self.learning_rate * grad_reg

        return param_updated


