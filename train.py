import wandb
import argparse
from model import FeedforwardNN
from optimizer import Optimizer
from data_loader import load_data
import numpy as np
from keras.utils import to_categorical # type: ignore
from loss_function import compute_loss
from sklearn.metrics import confusion_matrix


def train(model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size, loss_function):
    """
    Training the model
    """
    Y_train = to_categorical(y_train, num_classes=model.output_size)
    Y_val = to_categorical(y_val, num_classes=model.output_size)
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
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

            Y_pred = model.forward(X_batch)
            loss = compute_loss(Y_pred, Y_batch, loss_function)
            epoch_loss += loss

            grads = model.backward(Y_batch)
            model.update_parameters(grads, optimizer)

        avg_loss = epoch_loss / num_batches

        Y_val_pred = model.forward(X_val)
        val_loss = compute_loss(Y_val_pred, Y_val, loss_function)
        val_preds = np.argmax(Y_val_pred, axis=1)
        val_accuracy = np.mean(val_preds == y_val)

        Y_train_pred_full = model.forward(X_train)
        train_preds = np.argmax(Y_train_pred_full, axis=1)
        train_accuracy = np.mean(train_preds == y_train)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward NN on MNIST/Fashion-MNIST with Backpropagation")

    # WandB parameters
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname",
                        help="Project name used in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname",
                        help="WandB entity name")

    # Dataset selection
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"],
                        help="Dataset to use: 'mnist' or 'fashion_mnist'")

    # Training hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam",
                        choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"],
                        help="Optimizer type")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5,
                        help="Momentum for momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5,
                        help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5,
                        help="Beta1 for Adam and Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5,
                        help="Beta2 for Adam and Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6,
                        help="Epsilon for numerical stability in optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,
                        help="Weight decay regularization parameter")

    # Model architecture
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4,
                        help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128,
                        help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, default="tanh",
                        choices=["identity", "sigmoid", "tanh", "relu"],
                        help="Activation function for hidden layers")

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    run_name = (
        f"hl{config.num_layers}_sz{config.hidden_size}_b{config.batch_size}_"
        f"a_{config.activation}_lr{config.learning_rate}_o_{config.optimizer}"
    )
    wandb.run.name = run_name

    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)

    split_idx = int(X_train.shape[0] * 0.9)
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    hidden_sizes = [config.hidden_size] * config.num_layers

    input_size = 784
    output_size = 10

    model = FeedforwardNN(input_size, hidden_sizes, output_size,
                          activation=config.activation,
                          weight_init=config.weight_init)

    optimizer = Optimizer(
        optimizer_type=config.optimizer,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        beta=config.beta,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        weight_decay=config.weight_decay
    )

    train(model, optimizer, X_train, y_train, X_val, y_val,
          epochs=config.epochs, batch_size=config.batch_size, loss_function=config.loss)

    Y_test_pred = model.forward(X_test)
    test_preds = np.argmax(Y_test_pred, axis=1)
    test_accuracy = np.mean(test_preds == y_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    wandb.log({
        "test_accuracy": test_accuracy,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=Y_test_pred,
            y_true=y_test,
            class_names=[str(i) for i in range(output_size)]
        )
    })


    print(f"Test Accuracy: {test_accuracy:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()