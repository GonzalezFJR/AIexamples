# client.py

import time
import requests
import torch
import torch.nn as nn
import numpy as np

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

# Función para parsear el modelo
def parse_model(model):
    layers_data = []
    input_size = None

    # Obtener los módulos en orden
    modules = list(model.modules())[1:]  # Excluir el módulo raíz

    for layer in modules:
        if isinstance(layer, nn.Sequential):
            continue  # Saltamos contenedores secuenciales
        elif isinstance(layer, nn.Linear):
            layer_data = {
                'type': 'Linear',
                'neurons': layer.out_features,
                'weights': layer.weight.detach().numpy().tolist(),
                'biases': layer.bias.detach().numpy().tolist()
            }
            if input_size is None:
                input_size = layer.in_features
            layers_data.append(layer_data)
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
            activation = ''
            if isinstance(layer, nn.ReLU):
                activation = 'ReLU'
            elif isinstance(layer, nn.Sigmoid):
                activation = 'σ'
            elif isinstance(layer, nn.Tanh):
                activation = 'tanh'
            elif isinstance(layer, nn.Softmax):
                activation = 'softmax'

            layer_data = {
                'type': 'Activation',
                'activation': activation
            }
            layers_data.append(layer_data)
        else:
            # Manejar otros tipos si es necesario
            pass

    # Construir la estructura de capas
    layers = []

    # Capa de entrada
    layers.append({'neurons': input_size})

    i = 0
    while i < len(layers_data):
        layer_info = layers_data[i]
        if layer_info['type'] == 'Linear':
            neurons = layer_info['neurons']
            weights = layer_info['weights']
            biases = layer_info['biases']
            i += 1
            # Verificar si la siguiente capa es una función de activación
            activation_funcs = []
            if i < len(layers_data) and layers_data[i]['type'] == 'Activation':
                activation_func = layers_data[i]['activation']
                activation_funcs = [activation_func] * neurons
                i += 1
            else:
                activation_funcs = ['σ'] * neurons  # Valor por defecto
            # Agregar la capa
            layers.append({
                'neurons': neurons,
                'activations': activation_funcs,
                'weights': weights,
                'biases': biases
            })
        else:
            i += 1

    return layers

# Crear instancia del modelo
model = MyModel()

# Parsear el modelo
layers = parse_model(model)

# Lista de inputs de prueba
inputs_list = [
    [1.0, 0.5, -0.5],
    [0.0, -1.0, 2.0],
    [0.5, 0.5, 0.5],
    [2.0, -0.5, 1.0]
]

# Enviar la red y el primer input
url = 'http://localhost:8000/dnnvis/update_network'

data = {
    'layers': layers,
    'inputs': inputs_list[0]
}

response = requests.post(url, json=data)
print(response.json())

print("Primer input enviado. Observa la visualización.")
time.sleep(2)

# Enviar el segundo input
data['inputs'] = inputs_list[1]
response = requests.post(url, json=data)
print(response.json())

print("Segundo input enviado. Observa la visualización.")
time.sleep(2)

# Enviar el tercer input
data['inputs'] = inputs_list[2]
response = requests.post(url, json=data)
print(response.json())

print("Tercer input enviado. Observa la visualización.")
time.sleep(2)

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
layers2 = parse_model(model2)

# Enviar el nuevo modelo y un nuevo input
data = {
    'layers': layers2,
    'inputs': inputs_list[3]
}

response = requests.post(url, json=data)
print(response.json())

print("Estructura de la red modificada y nuevo input enviado. Observa la visualización.")
