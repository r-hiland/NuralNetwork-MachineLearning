### Neural Network from Scratch: MNIST Digit Classifier

This project involved implementing a forward feed neural network trained via stochastic gradient descent and backpropagation, using the MNIST dataset of handwritten digits.

The core functionality was developed from scratch in Python using NumPy and Pandas, with a focus on understanding and building the underlying mechanics of neural networks without the use of high-level libraries like TensorFlow or PyTorch.

**Key components:**

- **Activation Functions:** Implemented common activation functions (`tanh`, `sigmoid`, `relu`, `identity`) and their derivatives manually in a utility module (`math_util.py`) to support non-linear transformations.
  
- **Neural Network Architecture:**
  - Built a customizable feedforward neural network class in `nn.py`.
  - Implemented methods for adding layers, initializing weights, making predictions, and computing error through forward propagation.

- **Training:**
  - Developed the `fit` function to train the network using stochastic gradient descent and backpropagation.
  - Applied weight updates based on calculated gradients and error derivatives, enabling the network to learn from labeled examples.

- **Data Handling:**
  - Processed and shuffled the MNIST dataset (in CSV format), ensuring sample-label integrity throughout training and testing phases.

- **Performance:**
  - Achieved over **97% accuracy** on the MNIST validation set after training.
