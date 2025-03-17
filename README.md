# DA6401 Assignment 1
### By Akshay V me21b014

This repository contains all the code used for the assignment. It has been organised and works with the parameters specified in the question. 
The report containing the solutions are attached as a link below

### Solution Report

Link: [text](https://wandb.ai/retro-31-indian-institute-of-technology-madras/DL-assignment-01/reports/DA6401-Assignment-1-ME21B014--VmlldzoxMTY4OTkzMA)

## Project Structure

- **activations.py**  
  Contains activation functions (e.g., sigmoid, ReLU, tanh, identity) and their corresponding derivatives, along with helper functions to select the appropriate activation based on a string identifier.

- **data_loader.py**  
  Loads and preprocesses the dataset (MNIST or Fashion-MNIST). This module flattens and normalizes the images and, optionally, plots a sample image for each class for visualization purposes.
  `Note: To plot the sample images uncomment the plotting section in data_loader.py`

- **loss_function.py**  
  Implements loss functions (cross-entropy and mean squared error) to compute the loss between predicted probabilities and one-hot encoded true labels.

- **model.py**  
  Defines the `FeedforwardNN` class, which implements:
  - **Forward Propagation:** Computes activations for each layer, caches intermediate values, and applies softmax to the output layer.
  - **Backward Propagation:** Computes gradients for weights and biases using the chain rule.
  - **Parameter Update:** Delegates weight updates to the optimizer.

- **optimizer.py**  
  Contains the `Optimizer` class, which supports various optimization algorithms (SGD, momentum, Nesterov, RMSprop, Adam, Nadam) and incorporates weight decay (L2 regularization) into the update rules. It maintains state (such as momentum and adaptive learning rates) for optimizers that require it.

- **trainer.py**  
  Implements the training loop function. This function:


- **train.py**  
  Implements the training loop for each model configuration. It:

  - Shuffles the training data and divides it into mini-batches.
  - Performs forward and backward passes.
  - Updates the model parameters.
  - Computes and logs metrics such as training loss, training accuracy, validation loss, and validation accuracy.
  - Parses command-line arguments for model architecture, training hyperparameters, optimizer settings, and wandb configuration.
  - Initializes wandb and assigns a descriptive run name based on the hyperparameter configuration.
  - Loads the dataset and creates a validation split.
  - Instantiates the neural network model and optimizer.
  - Executes the training loop and, after training, evaluates the model on the test set, logging the test accuracy.

- **sweep_config.yaml**  
  Contains the configuration for wandb hyperparameter sweeps. This file defines the search space for hyperparameters and the optimization strategy. During a sweep.

You can install the required packages using:

```bash
pip install requirements.txt
```
## Running the Project

To train the model using default settings, execute:
```bash
python train.py -wp sample -we retro-31-indian-institute-of-technology-madras -d fashion_mnist -e 10 -b 64 -l cross_entropy -o nadam -lr 0.001 -m 0.5 -beta 0.5 -beta1 0.5 -beta2 0.5 -eps 1e-6 -w_d 0.0 -w_i xavier -nhl 4 -sz 64 -a tanh
```
## Hyperparameter Sweeps

The project supports hyperparameter sweeps via wandb. The sweep_config.yaml file defines the search space and optimization strategy. Each run in the sweep automatically receives a descriptive name based on its hyperparameters, which allows for easy comparison of performance metrics across runs on the wandb dashboard.