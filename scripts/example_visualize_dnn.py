# client.py

import time
import requests
import torch
import torch.nn as nn

from api_visual_dnn import draw_network

# Definir el modelo de PyTorch
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


# Crear instancia del modelo
model = MyModel()

# Lista de inputs de prueba
inputs_list = [
    [1.0, 0.5, -0.5],
    [0.0, -1.0, 2.0],
    [0.5, 0.5, 0.5],
    [2.0, -0.5, 1.0]
]

for input in inputs_list:
    draw_network(model, input)
    time.sleep(1)

# Modificar la estructura de la red
# Creamos un nuevo modelo con una capa adicional
class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 5),
            nn.Tanh(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

# Parsear el nuevo modelo
model2 = MyModel2()

for input in inputs_list:
    draw_network(model2, input)
    time.sleep(1)