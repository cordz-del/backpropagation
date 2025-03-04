import numpy as np

# Activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Initialize network parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01  # Hidden layer weights
    b1 = np.zeros((n_h, 1))                # Hidden layer biases
    W2 = np.random.randn(n_y, n_h) * 0.01   # Output layer weights
    b2 = np.zeros((n_y, 1))                # Output layer biases
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)                      # Activation for hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)                      # Activation for output layer
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Compute binary cross-entropy loss
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return np.squeeze(cost)

# Backward propagation
def backward_propagation(X, Y, cache, W2):
    m = X.shape[1]
    (Z1, A1, Z2, A2) = cache
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # Derivative of tanh is 1 - tanh^2
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Update parameters using gradient descent
def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    W1 -= learning_rate * gradients["dW1"]
    b1 -= learning_rate * gradients["db1"]
    W2 -= learning_rate * gradients["dW2"]
    b2 -= learning_rate * gradients["db2"]
    return W1, b1, W2, b2

# Training loop for the two-layer neural network
def train_neural_network(X, Y, n_h, num_iterations=5000, learning_rate=0.01, print_cost=False):
    n_x = X.shape[0]  # Number of input features
    n_y = Y.shape[0]  # Number of outputs
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        gradients = backward_propagation(X, Y, cache, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Example usage:
if __name__ == "__main__":
    np.random.seed(1)
    # Create synthetic data: 2 features, 400 examples
    X = np.random.randn(2, 400)
    # Binary labels: 1 if sum of features > 0, else 0
    Y = (np.sum(X, axis=0) > 0).astype(int).reshape(1, 400)
    
    # Train the network with 4 hidden neurons
    parameters = train_neural_network(X, Y, n_h=4, num_iterations=5000, learning_rate=0.01, print_cost=True)
