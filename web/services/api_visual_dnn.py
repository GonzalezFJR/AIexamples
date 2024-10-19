'''
 API para visualizar una red neuronal en tiempo real

'''

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
from fastapi import APIRouter

import numpy as np
import time
import requests

'''
 Funciones para parsear el modelo
'''

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

def draw_network(model, inputs, ip='localhost', port=8000, route='/dnnvis/update_network'):
    # Código para visualizar la red
    layers = parse_model(model)
    url = f'http://{ip}:{port}{route}'
    data = {
        'layers': layers,
        'inputs': inputs
    }
    response = requests.post(url, json=data)
    return response.json()


'''
 API
'''

dnnvisrouter = APIRouter(prefix="/dnnvis")

# Definir el modelo de datos para la red y los inputs
class NetworkData(BaseModel):
    layers: list
    inputs: List[float]

# Variable global para almacenar el estado de la red
network_state = {
    'layers': [],
    'inputs': []
}

# Lista para almacenar los WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Enviar el estado actual de la red al conectarse
        await self.send_network_state(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_network_state(self, websocket: WebSocket):
        await websocket.send_json({
            'type': 'network_state',
            'layers': network_state['layers'],
            'inputs': network_state['inputs']
        })

    async def broadcast_network_state(self):
        for connection in self.active_connections:
            await self.send_network_state(connection)

manager = ConnectionManager()

# Ruta para obtener la red actual (ya no necesaria, pero la dejamos por compatibilidad)
@dnnvisrouter.get('/network')
async def get_network():
    return JSONResponse(content={'layers': network_state['layers']})

# Ruta para obtener los inputs actuales (ya no necesaria)
@dnnvisrouter.get('/inputs')
async def get_inputs():
    return JSONResponse(content={'inputs': network_state['inputs']})

# Ruta para actualizar la red y los inputs
@dnnvisrouter.post('/update_network')
async def update_network(data: NetworkData):
    network_state['layers'] = data.layers
    network_state['inputs'] = data.inputs
    # Notificar a todos los clientes conectados
    await manager.broadcast_network_state()
    return {'status': 'Network updated successfully'}

# Endpoint de WebSocket
@dnnvisrouter.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Esperar a que el cliente envíe algo (mantiene la conexión abierta)
            data = await websocket.receive_text()
            # No hacemos nada con los datos recibidos por ahora
    except WebSocketDisconnect:
        manager.disconnect(websocket)
