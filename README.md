# tinygrad : Improved Micrograd Implementation

This project is an enhanced version of the micrograd neural network library, inspired by Andrej Karpathy's micrograd. It provides a simple, educational implementation of backpropagation and neural networks from scratch in Python.

## New Features and Improvements

1. **Extended Activation Functions**: In addition to ReLU, we've implemented sigmoid and tanh activation functions.

2. **Type Hints**: Added type annotations to improve code readability and catch potential errors early.

3. **Mini-batch Processing**: The MLP class now supports processing multiple samples at once, improving training efficiency.

4. **Training Loop**: Implemented a basic training function with customizable learning rate and number of epochs.

5. **Regularization**: Added L1 and L2 regularization options to prevent overfitting.

6. **Dropout**: Implemented dropout layers for better generalization.

7. **Enhanced Visualization**: Improved the `draw_dot` function to provide more detailed information in the computation graph visualization.

## Project Structure

The project consists of three main Python files:

1. `engine.py`: Contains the core `Value` class for automatic differentiation.
2. `nn.py`: Implements neural network components (Neuron, Layer, MLP) and the training function.
3. `tracing.py`: Provides functionality for visualizing the computation graph.

## Usage

Here's a quick example of how to use this improved micrograd implementation:

```python
import nn
from engine import Value

# Create a simple neural network
model = nn.MLP(2, [4, 4, 1])

# Create input data
X = [[Value(1.0), Value(-2.0)], [Value(-1.0), Value(2.0)]]
y = [Value(1.0), Value(-1.0)]

# Train the model
nn.train(model, X, y, learning_rate=0.1, epochs=100, l1_lambda=0.01, l2_lambda=0.01)

# Make a prediction
x_test = [Value(0.5), Value(-1.0)]
y_pred = model.forward(x_test)

print(f"Prediction: {y_pred.data}")
```

To visualize the computation graph:

```python
from tracing import draw_dot

dot = draw_dot(y_pred)
dot.render('computation_graph', view=True)
```

## Requirements

- Python 3.6+
- graphviz (for visualization)

Install the required packages using:

```
pip install graphviz
```

## Contributing

Contributions to improve the implementation or add new features are welcome! Please feel free to submit a pull request or open an issue for discussion.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

This project is inspired by Andrej Karpathy's micrograd. The improvements and extensions aim to make it more feature-rich while maintaining its educational value.
