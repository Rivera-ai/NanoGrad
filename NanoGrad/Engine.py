import math
import numpy as np

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def tanh(self):
        x = self.data
        if x < -20:
            t = -1
        elif x > 20:
            t = 1
        else:
            e2x = math.exp(2 * x)
            t = (e2x - 1) / (e2x + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def __getstate__(self):
        # Excluir el _backward al serializar
        state = self.__dict__.copy()
        del state['_backward']
        return state

    def __setstate__(self, state):
        # Restaurar el estado sin _backward
        self.__dict__.update(state)
        self._backward = lambda: None

class Tensor:
    def __init__(self, data, requires_grad=False):
        """ Implementación del Tensor usando Numpy
        Parametros:
        - data: un array de numpy es la data del tensor.
        - requires_grad: dato tipo booleano por si se necesitan gradientes para este tensor. 
        """

        self.data = np.array(data, dtype=np.float32) # Maybe talvez despues lo cambie a float16 por cuestion de eficiencia
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = []
        self._op = ''

    def __repr__(self): # Imprimir la representación del Tensor ejemplo: Tensor(data=[[1, 2], [3, 4]], grad=[[0.1, 0.2], [0.3, 0.4]])
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self):
        """Reliza el backward en el grafo computacional """
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for parent in t._prev:
                    build_topo(parent)
                topo.append(t)
        build_topo(self)

        self.grad = np.ones_like(self.data) # Inicializamos gradiente como 1 (derivada de sí mismo)

        for t in reversed(topo):
            t._backward()

    def __add__(self, other):
        """ Suma de tensores """

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward
        other._prev = [self, other]
        out._op = '+'
        return out

    def __mul__(self, other):
        """ Multiplicación elemento a elemento"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        
        out._backward = _backward
        out._prev = [self, other]
        out._op = '*'
        return out

    def matmul(self, other):
        """Producto matricial entre dos tensores"""

        assert self.data.shape[-1] == other.data.shape[-2], "Dimensiones incompatibles para multiplicación de matrices."
        out = Tensor(np.dot(self.data, other.data), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)
        
        out._backward = _backward
        out._prev = [self, other]
        out._op = 'matmul'

        return out

    def relu(self):
        """Función de activación ReLU."""
        out = Tensor(np.maximum(0, self.data), requires_grad=True)

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        out._prev = [self]
        out._op = 'relu'
        return out

    def sum(self, axis=None):
        """Suma de todos los elementos o en un eje específico."""
        out = Tensor(np.sum(self.data, axis=axis), requires_grad=True)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._prev = [self]
        out._op = 'sum'
        return out

    def T(self):
        """Transposición del tensor."""
        out = Tensor(self.data.T, requires_grad=True)

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        out._prev = [self]
        out._op = 'transpose'
        return out

