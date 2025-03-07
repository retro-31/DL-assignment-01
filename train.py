import argparse
from model import FeedforwardNN
from optimizer import Optimizer
from data_loader import load_data
import numpy as np
from keras.utils import to_categorical # type: ignore
from loss_function import compute_loss

def train(model, optimizer, X_train, y_train, epochs, batch_size, loss_function):
    """
    Training loop for the neural network.
    """
    # One-hot encode labels
    Y_train = to_categorical(y_train, num_classes=model.output_size)
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[permutation]
        Y_shuffled = Y_train[permutation]
        
        epoch_loss = 0
        num_batches = num_samples // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]
            
            # Forward pass
            Y_pred = model.forward(X_batch)
            loss = compute_loss(Y_pred, Y_batch, loss_function)
            epoch_loss += loss
            
            # Backward pass and parameter update
            grads = model.backward(Y_batch)
            model.update_parameters(grads, optimizer)
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward NN on MNIST/Fashion-MNIST with Backpropagation")
    
    # WandB parameters (for potential integration)
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name used in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="WandB entity name")
    
    # Dataset selection
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use: 'mnist' or 'fashion_mnist'")
    
    # Training hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer type")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam and Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam and Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for numerical stability in optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay regularization parameter")
    
    # Model architecture
    parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices=["random", "xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function for hidden layers")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Loading dataset
    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)
    
    # hidden layers configuration
    hidden_sizes = [args.hidden_size] * args.num_layers
    print(f"Using {args.num_layers} hidden layers with {args.hidden_size} neurons each: {hidden_sizes}")
    
    # input and output dimensions
    input_size = 784
    output_size = 10
    
    # Initializing the neural network
    model = FeedforwardNN(input_size, hidden_sizes, output_size,
                          activation=args.activation,
                          weight_init=args.weight_init)
    
    # Initializing the optimizer
    optimizer = Optimizer(
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay
    )
    
    # training
    train(model, optimizer, X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, loss_function=args.loss)
    
if __name__ == "__main__":
    main()