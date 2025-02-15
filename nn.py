import random
from engine import Value    # Importing the Value class from engine

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x):
        act = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)  # Weighted sum
        return act.tanh()  # Activation function
    
    def parameters(self):
        return self.w + [self.b]  # Ensuring it's not a mutable reference

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # Passing through each layer
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
