from NanoGrad.Engine import Value
from NanoGrad.nn import *
from NanoGrad.Checkpoint import SaveCheckpoint, LoadCheckpoint
import math
import random

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

model = MLP(3, [16, 16, 1])  
checkpoint = LoadCheckpoint(model, 'model.pkl')
checkpoint.load()

def predict(model, x, X_orig):
    x_norm = []
    for i, xi in enumerate(x):
        col = [row[i].data for row in X_orig]
        mean = sum(col) / len(col)
        std = math.sqrt(sum((x - mean)**2 for x in col) / len(col))
        x_norm.append((xi - mean) / (std + 1e-8))
    return model([Value(xi) for xi in x_norm]).data * 100000  # Desescalamos el resultado

# Ejemplo de predicción
nueva_casa = [170, 6, 2]
precio_predicho = predict(model, nueva_casa, X)
print(f"Precio predicho para una casa de {nueva_casa[0]}m2, {nueva_casa[1]} habitaciones y {nueva_casa[2]} baños: ${precio_predicho:.2f}")

