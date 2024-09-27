from NanoGrad.Engine import Tensor
from NanoGrad.nn import MLP
from NanoGrad.nn import Linear
import numpy as np

# Crear una red MLP con 3 capas: input=2, hidden=3, output=1
mlp = MLP([2, 3, 1])

# Entrenamiento simple de demostración
x = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)  # Dos ejemplos de entrada
y_true = Tensor(np.array([[1], [0]]), requires_grad=True)  # Verdaderos valores de salida

# Hacer una predicción
mlp.zero_grad()
y_pred = mlp(x)

# Calcular el error cuadrático medio (MSE Loss)
loss = ((y_true - y_pred) ** 2).sum()

print("Forma de y_true:", y_true.data.shape)
print("Forma de y_pred:", y_pred.data.shape)
print("Pérdida inicial:", loss.data)

# Imprimir valores antes de la retropropagación
print("Valores antes de la retropropagación:")
for i, p in enumerate(mlp.parameters()):
    print(f"Parámetro {i}:")
    print(p.data)

# Backward para calcular gradientes
loss.backward()

print("Gradiente de la pérdida:", loss.grad)

# Ver los gradientes de cada tensor en las capas
print("Gradientes después de la retropropagación:")
for i, p in enumerate(mlp.parameters()):
    print(f"Gradiente de parámetro {i}:")
    print(p.grad)
    # Verificar si hay NaN o Inf en los gradientes
    if not np.all(np.isfinite(p.grad)):
        print(f"¡Advertencia! Gradiente no finito en parámetro {i}")

# Imprimir los gradientes de los parámetros de cada capa
for i, layer in enumerate(mlp.layers):
    print(f"Gradientes de la capa {i}:")
    print(f"Gradiente de W: {layer.W.grad}")
    print(f"Gradiente de b: {layer.b.grad}")