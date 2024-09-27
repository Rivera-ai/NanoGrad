from NanoGrad.Engine import Value
from NanoGrad.nn import *
from NanoGrad.Checkpoint import SaveCheckpoint, LoadCheckpoint
import math
import random

def train(model, X, y, learning_rate=0.0001, epochs=10000):
    for epoch in range(epochs):
        total_loss = 0
        for xi, yi in zip(X, y):
            y_pred = model(xi)
            loss = mse_loss(yi, y_pred)
            total_loss += loss.data

            model.zero_grad()
            loss.backward()

            for p in model.parameters():
                p.data -= learning_rate * p.grad

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")

# Preparar los datos (ejemplo ampliado)
X = [
    [Value(100), Value(2), Value(1)],
    [Value(150), Value(3), Value(2)],
    [Value(80), Value(2), Value(1)],
    [Value(120), Value(2), Value(2)],
    [Value(160), Value(3), Value(2)],
    [Value(90), Value(1), Value(1)],
    [Value(110), Value(2), Value(1)],
    [Value(140), Value(3), Value(2)],
    [Value(130), Value(2), Value(2)],
    [Value(170), Value(4), Value(3)],
]
# Escalamos los precios dividiéndolos por 100000
y = [Value(2), Value(3), Value(1.8), Value(2.2), Value(3.2),
     Value(1.7), Value(2.1), Value(2.8), Value(2.5), Value(3.5)]

# Normalizar los datos
X_norm = normalize(X)

# Entrenar la red
model = MLPV(3, [16, 16, 1])  # 3 entradas, dos capas ocultas de 16 neuronas, 1 salida
train(model, X_norm, y, learning_rate=0.0001, epochs=5000)
checkpoint = SaveCheckpoint(model, 'model.pkl')
checkpoint.save()

# Hacer predicciones
def predict(model, x, X_orig):
    x_norm = []
    for i, xi in enumerate(x):
        col = [row[i].data for row in X_orig]
        mean = sum(col) / len(col)
        std = math.sqrt(sum((x - mean)**2 for x in col) / len(col))
        x_norm.append((xi - mean) / (std + 1e-8))
    return model([Value(xi) for xi in x_norm]).data * 100000  # Desescalamos el resultado

# Ejemplo de predicción
nueva_casa = [120, 2, 2]
precio_predicho = predict(model, nueva_casa, X)
print(f"Precio predicho para una casa de {nueva_casa[0]}m2, {nueva_casa[1]} habitaciones y {nueva_casa[2]} baños: ${precio_predicho:.2f}")
