from NanoGrad.Engine import Tensor
from NanoGrad.nn import *
from NanoGrad.Checkpoint import SaveCheckpoint, LoadCheckpoint
import math
import random
import numpy as np

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum()

def train(model, X, y, learning_rate=0.0001, epochs=10000):
    for epoch in range(epochs):
        total_loss = 0
        for xi, yi in zip(X, y):
            xi_tensor = Tensor([x.data for x in xi], requires_grad=False)
            y_pred = model(xi_tensor)
            loss = mse_loss(yi, y_pred)
            total_loss += loss.data

            model.zero_grad()
            loss.backward()

            for p in model.parameters():
                p.data -= learning_rate * p.grad

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")


# Preparar los datos
X = [
    [Tensor((100, 2, 1), requires_grad=False)],
    [Tensor((150, 3, 1), requires_grad=False)],
    [Tensor((80, 2, 1), requires_grad=False)],
    [Tensor((120, 2, 2), requires_grad=False)],
    [Tensor((160, 3, 2), requires_grad=False)],
    [Tensor((90, 1, 1), requires_grad=False)],
    [Tensor((110, 2, 1), requires_grad=False)],
    [Tensor((140, 3, 2), requires_grad=False)],
    [Tensor((130, 2, 2), requires_grad=False)],
    [Tensor((170, 4, 3), requires_grad=False)],
]

# Escalamos los precios dividiéndolos por 100000
y = [Tensor((2, 3, 1.8), requires_grad=True), 
     Tensor((2.2, 3.2), requires_grad=True),
     Tensor((1.7, 2.1), requires_grad=True), 
     Tensor((2.8, 2.5), requires_grad=True), 
     Tensor(3.5, requires_grad=True)]





# Entrenar la red
model = MLP([3, 16, 16, 1])  # 3 entradas, dos capas ocultas de 16 neuronas, 1 salida
X_norm = TensorNormalize(X)
train(model, X_norm, y, learning_rate=0.0001, epochs=10)

def predict(model, x, X_orig):
    x_norm = []
    
    for i, xi in enumerate(x):
        # Handle nested list structure: extract tensor values directly
        col = [row[0].data[i] for row in X_orig]  # Access the i-th element of the tensor in X_orig

        # Handle cases where tensors might be multi-dimensional
        col = np.array([float(c.flatten()[0]) if isinstance(c, (list, np.ndarray)) else float(c) for c in col])

        mean = np.mean(col)
        std = np.std(col)

        # Normalize input xi
        x_norm.append((xi - mean) / (std + 1e-8))

    # Convert normalized input to Tensor
    x_tensor = Tensor(x_norm, requires_grad=False)
    
    # Predict and scale back to original price range
    return model(x_tensor).data * 100000

# Ejemplo de predicción
nueva_casa = [120, 2, 2]
precio_predicho = predict(model, nueva_casa, X)
print(f"Precio predicho para una casa de {nueva_casa[0]}m2, {nueva_casa[1]} habitaciones y {nueva_casa[2]} baños: ${float(precio_predicho[0][0]):.2f}")

