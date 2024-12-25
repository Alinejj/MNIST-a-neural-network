
# Neural Network for Handwritten Digit Classification

This project implements a neural network from scratch, without using popular deep learning libraries like TensorFlow or PyTorch. The goal is to classify handwritten digits from the MNIST dataset, showcasing a deep understanding of neural network architecture and optimization techniques.

# Features:
- Dataset: The neural network uses the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits (0-9).

- Implementation: Built entirely using Python, NumPy, and mathematical operations.

- Architecture:
     - Input layer: 784 nodes (28x28 pixel images flattened into vectors)
  - Hidden layer: Configurable number of nodes with ReLU activation
  - Output layer: 10 nodes (one for each digit) with softmax activation

- Optimization: Implements forward propagation, backward propagation, and gradient descent manually.

- Accuracy: Achieved 84% accuracy on the MNIST test set.

# Structure
- Data Preprocessing: The MNIST dataset is normalized and split into training and test sets.
- Forward Propagation: Computes the outputs for the hidden and output layers.
- Backward Propagation: Calculates gradients for weights and biases to minimize the error.
- Gradient Descent: Updates weights and biases using the gradients and a learning rate.
- Evaluation: Measures the model’s performance in terms of accuracy.

# How It Works
1.  Initialization:
- Randomly initializes weights and biases for all layers.
2. Forward Propagation:
- Computes the weighted sum of inputs and applies activation functions (ReLU and softmax).
3. Backward Propagation:
- Computes gradients of loss with respect to weights and biases.
4. Training:
- Updates parameters using gradient descent over multiple iterations to minimize the error.
5. Evaluation:
- Tests the model on the test dataset and calculates accuracy.
