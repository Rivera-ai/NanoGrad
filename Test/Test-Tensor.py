from NanoGrad.Engine import *

# Crear dos tensores
a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[5, 6], [7, 8]], requires_grad=True)

# Producto matricial
c = a.matmul(b)

# Suma de todos los elementos
d = c.sum()

# Realizar la retropropagación
d.backward()

print(a)  # Tensor con los gradientes calculados
print(b)  # Tensor con los gradientes calculados
