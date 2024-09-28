import math
import numpy as np
import random

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
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = []
        self._op = ''

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
            print(f"Gradiente inicializado para {self}: {self.grad}")

    def backward(self):
        self._ensure_grad()

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for parent in t._prev:
                    build_topo(parent)
                print(f"Registrando operación: {t._op}")
                topo.append(t)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        print(f"Inicializando gradiente en backward: {self.grad}")
        for t in reversed(topo):
            print(f"Backward para operación: {t._op}")
            print(f"Gradiente antes: {t.grad}")
            t._backward()
            print(f"Gradiente después: {t.grad}")

    def __add__(self, other): 


        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            print(f"Backward de la suma: grad de salida (out): {out.grad}")

            def apply_grad(tensor, grad):

            # Si el gradiente es un escalar o un array de un solo elemento,
            # lo sumamos directamente sin reshape
                if np.isscalar(grad) or grad.size == 1:
                    tensor.grad += np.sum(grad)
                else:
                # Expandir dimensiones si es necesario
                    while tensor.grad.ndim < grad.ndim:
                        tensor.grad = np.expand_dims(tensor.grad, axis=-1)
                    while grad.ndim < tensor.grad.ndim:
                        grad = np.expand_dims(grad, axis=-1)

                    if tensor.grad.shape != grad.shape:
                    # Realizar suma reducida en los ejes necesarios
                        axes = tuple(i for i in range(grad.ndim) if tensor.grad.shape[i] == 1 and grad.shape[i] != 1)
                        grad = np.sum(grad, axis=axes, keepdims=True)
                
                    tensor.grad += grad

            if self.requires_grad:
                self._ensure_grad()
                grad_self = out.grad

                apply_grad(self, grad_self)

                print(f"Gradiente self después de acumular: {self.grad}")

            if other.requires_grad:
                other._ensure_grad()
                grad_other = out.grad

                apply_grad(other, grad_other)
                print(f"Gradiente other después de acumular: {other.grad}")

        out._backward = _backward
        out._prev = [self, other]
        out._op = '+'
        return out

    def __sub__(self, other):
        self._ensure_grad()
        return self + (-other)

    def __mul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            print(f"Backward de la multiplicación: grad de salida (out): {out.grad}")
            if self.requires_grad:
                self._ensure_grad()
                print(f"Gradiente self antes de acumular: {self.grad}")
                self.grad += np.broadcast_to(other.data * out.grad, self.data.shape)
                print(f"Gradiente self después de acumular: {self.grad}")
            if other.requires_grad:
                other._ensure_grad()
                print(f"Gradiente other antes de acumular: {other.grad}")
                other.grad += np.broadcast_to(self.data * out.grad, other.data.shape)
                print(f"Gradiente other después de acumular: {other.grad}")

        out._backward = _backward
        out._prev = [self, other]
        out._op = '*'
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "El exponente debe ser un número (int o float)"
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._ensure_grad()  # Asegurar inicialización
                grad = exponent * self.data ** (exponent - 1) * out.grad
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        out._prev = [self]
        out._op = f'**{exponent}'
        return out

    def matmul(self, other):


        assert self.data.shape[-1] == other.data.shape[-2], "Dimensiones incompatibles para multiplicación de matrices."
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.dot(self.data, other.data), requires_grad=requires_grad)

        def _backward():
            print(f"Backward para matmul")
            if self.requires_grad:
                self._ensure_grad()  # Asegurar inicialización
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other._ensure_grad()  # Asegurar inicialización
                other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        out._prev = [self, other]
        out._op = 'matmul'

        return out

    def relu(self):

        out = Tensor(np.maximum(0, self.data), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (self.data > 0) * out.grad
                print(f"Gradiente ReLU: {self.grad}")

        out._backward = _backward
        out._prev = [self]
        out._op = 'relu'
        return out

    def leaky_relu(self, alpha=0.01):
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), requires_grad=True)

        def _backward():
            print(f"Backward de Leaky ReLU: grad de salida (out): {out.grad}")
            if self.requires_grad:
                self._ensure_grad()  # Asegurar inicialización
                self.grad += np.where(self.data > 0, 1, alpha) * out.grad

        out._backward = _backward
        out._prev = [self]
        out._op = 'leaky_relu'
        return out


    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        def _backward():
            print(f"Backward para sum: grad de salida (out): {out.grad}")
            if self.requires_grad:
                self._ensure_grad()  # Asegurar inicialización
                grad = out.grad
                # Si axis está especificado, expandir grad para que coincida con self.data.shape
                if axis is not None:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    grad = np.expand_dims(grad, axes)
                # Difundir grad para que coincida con self.data.shape
                grad = np.broadcast_to(grad, self.data.shape)
                print(f"Gradiente self antes de acumular: {self.grad}")
                self.grad += grad
                print(f"Gradiente self después de acumular: {self.grad}")

        out._backward = _backward
        out._prev = [self]
        out._op = 'sum'
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += -out.grad  # La derivada de -x con respecto a x es -1

        out._backward = _backward
        out._prev = [self]
        out._op = 'neg'
        return out

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

