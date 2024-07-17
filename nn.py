import random
from typing import List, Union
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin: int, activation: str = 'relu'):
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0)
        self.activation = activation

    def __call__(self, x: List[Union[float, Value]]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'tanh':
            return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize()}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):
        if self.training:
            mask = [random.random() > self.p for _ in range(len(x))]
            return [xi * mi / (1 - self.p) if mi else 0 for xi, mi in zip(x, mask)]
        return x

class MLP(Module):
    def __init__(self, nin, nouts, dropout_p=0.5):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            self.layers.append(Layer(sz[i], sz[i+1], activation='relu' if i != len(nouts)-1 else 'linear'))
            if i != len(nouts) - 1:  # Don't add dropout after the last layer
                self.layers.append(Dropout(dropout_p))

    def __call__(self, X):
        # X is now a list of input samples
        return [self.forward(x) for x in X]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

def train(model, X, y, learning_rate=0.01, epochs=100, l1_lambda=0, l2_lambda=0):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = sum((y_pred[i] - y[i])**2 for i in range(len(y))) / len(y)
        
        # Add regularization to the loss
        l1_reg = sum(abs(p.data) for p in model.parameters())
        l2_reg = sum(p.data**2 for p in model.parameters())
        loss += l1_lambda * l1_reg + l2_lambda * l2_reg
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update parameters
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.data}')
