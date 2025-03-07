import argparse
from model import FeedforwardNN
from data_loader import load_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Train a Feedforward NN on Fashion-MNIST")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1,
                        help="Number of hidden layers in the network (used for validation with -sz)")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4,
                        help="number of neurons in each hidden layer.")
    parser.add_argument("-a", "--activation", type=str, default="sigmoid",
                        choices=["sigmoid", "relu", "tanh", "identity"],
                        help="Activation function for hidden layers")
    parser.add_argument("-wi", "--weight_init", type=str, default="random",
                        choices=["random", "xavier"],
                        help="Weight initialization method")
    
    args = parser.parse_args()
    
    hidden_sizes = [args.hidden_size] * args.num_layers
    
    # input and output sizes
    input_size = 784
    output_size = 10
    
    # feedforward network initialization
    nn = FeedforwardNN(input_size, hidden_sizes, output_size, activation=args.activation, weight_init=args.weight_init)
    (X_train, y_train), (X_test, y_test) = load_data()
    
    # run on batch of 5
    predictions = nn.forward(X_train[:5])
    
    print("Predicted probabilities for the dummy batch:")
    print(predictions)

if __name__ == "__main__":
    main()
