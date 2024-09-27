from NanoGrad.Engine import Value, Tensor
import random
import math
import numpy as np

class Neuron:
    def __init__(self, nin, activation='relu'):
        self.w = [Value(random.uniform(-1/math.sqrt(nin), 1/math.sqrt(nin))) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.activation == 'relu' else act

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, activation='relu'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLPV:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation='relu') for i in range(len(nouts)-1)]
        self.layers.append(Layer(sz[-2], sz[-1], activation='linear'))  # Linear output layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

def mse_loss(y_true, y_pred):
    return (y_true - y_pred)**2

def normalize(X):
    X_norm = []
    for i in range(len(X[0])):
        col = [x[i].data for x in X]
        mean = sum(col) / len(col)
        std = math.sqrt(sum((x - mean)**2 for x in col) / len(col))
        X_norm.append([(x - mean) / (std + 1e-8) for x in col])
    return [[Value(x[i]) for i in range(len(X[0]))] for x in zip(*X_norm)]


def TensorNormalize(X):
    X_data = np.array([[x.data for x in row] for row in X])
    mean = X_data.mean(axis=0)
    std = X_data.std(axis=0)
    X_norm = (X_data - mean) / (std + 1e-8)
    return [[Tensor(x, requires_grad=False) for x in row] for row in X_norm]


class Linear:
    def __init__(self, in_features, out_features, requires_grad=True):
        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.W._ensure_grad()
        self.b._ensure_grad()

    def __call__(self, x):
        print(f" Pre matmul: x: {x}, W: {self.W}, b: {self.b}")
        out = x.matmul(self.W) + self.b
        print(f" Despues de matmul: x: {x}, W: {self.W}, b: {self.b}")
        print(f"Linear layer output: {out.data}")
        return out

    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)

class MLP:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Linear(layers[i], layers[i+1]))

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = x.leaky_relu()
                print(f"LeakyReLU output at layer {i}: {x.data}")
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()