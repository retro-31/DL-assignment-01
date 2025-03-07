import numpy as np

def compute_loss(Y_pred, Y_true, loss_function):
    """
    Computes the loss based on the specified loss function.
    """
    m = Y_true.shape[0]
    if loss_function == "cross_entropy":
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
    elif loss_function == "mean_squared_error":
        loss = np.sum((Y_pred - Y_true) ** 2) / (2 * m)
    else:
        raise ValueError("Unsupported loss function.")
    return loss