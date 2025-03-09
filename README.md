# backpropagation
Understanding and implementing backpropagation in Neural Networks

## Implementation Details

This repository contains a Python script that implements a simple two-layer neural network from scratch using NumPy. The implementation includes the following functions:

- `initialize_parameters`: Initializes the weights and biases for the neural network.
- `forward_propagation`: Performs forward propagation through the neural network to compute the output.
- `compute_cost`: Computes the cost (or loss) of the neural network's predictions.
- `backward_propagation`: Performs backpropagation to compute the gradients of the cost with respect to the weights and biases.
- `update_parameters`: Updates the weights and biases using gradient descent.

## Usage

To run the script, simply execute it using Python:

```bash
python backpropagation.py
```

The script will output the cost at each iteration and the final predictions of the neural network.

## Certification Relevance

This repository is relevant to the "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization" certification. It demonstrates the implementation of backpropagation in neural networks, which is a fundamental concept in deep learning. The script provides an understanding of how to optimize neural networks by tuning hyperparameters and applying regularization techniques.
