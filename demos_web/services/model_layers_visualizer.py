import torch.nn as nn
import json
import requests
import torch

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

import uvicorn


'''
 Funciones para parsear el modelo y para pasarlo al visualizador web con un POST
'''

def parse_model(model, input_shape):
    layers = []
    def hook_fn(module, input, output):
        layer = {}
        layer['type'] = module.__class__.__name__

        # Manejar salidas de RNN que son tuplas
        if isinstance(output, tuple):
            output = output[0]

        # Convertir tamaños de tensores a listas
        layer['input_shape'] = list(input[0].size())
        layer['output_shape'] = list(output.size())

        # Extraer parámetros específicos según el tipo de capa
        if isinstance(module, nn.Conv2d):
            layer['kernel_size'] = module.kernel_size
            layer['stride'] = module.stride
            layer['padding'] = module.padding
            layer['in_channels'] = module.in_channels
            layer['out_channels'] = module.out_channels
            layer['num_filters'] = module.out_channels
        elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
            layer['kernel_size'] = module.kernel_size
            layer['stride'] = module.stride
            layer['padding'] = module.padding
        elif isinstance(module, nn.Linear):
            layer['in_features'] = module.in_features
            layer['out_features'] = module.out_features
        elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            layer['input_size'] = module.input_size
            layer['hidden_size'] = module.hidden_size
            layer['num_layers'] = module.num_layers
            layer['bidirectional'] = module.bidirectional
        # Puedes agregar más tipos de capas según sea necesario
        layers.append(layer)

    hooks = []
    for module in model.modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(hook_fn))

    # Generar una entrada aleatoria con la forma adecuada
    x = torch.randn(input_shape)
    model.eval()
    with torch.no_grad():
        model(x)

    # Remover los hooks
    for hook in hooks:
        hook.remove()

    return layers

def visualize(model, input_shape, ip='localhost', port=8000):
    layers = parse_model(model, input_shape)
    model_json = json.dumps(layers)

    # Construir la URL del servidor
    url = f'http://{ip}:{port}/upload_model_layer_visualizer'
    headers = {'Content-Type': 'application/json'}

    # Enviar el JSON al servidor
    response = requests.post(url, data=model_json, headers=headers)

    if response.status_code == 200:
        print("Modelo enviado exitosamente al servidor.")
    else:
        print(f"Error al enviar el modelo: {response.text}")

'''
 API y rutas
'''

layervisrouter = APIRouter()

model_data = None  # Variable global para almacenar el JSON del modelo
model_version = 0  # Nueva variable para el control de versiones

@layervisrouter.post("/upload_model_layer_visualizer")
async def upload_model(request: Request):
    global model_data, model_version
    model_data = await request.json()
    model_version += 1  # Incrementar la versión
    return {"status": "Modelo recibido exitosamente"}


@layervisrouter.get("/get_model_layer_visualizer")
async def get_model():
    if model_data is not None:
        return JSONResponse(content=model_data)
    else:
        return {"error": "No hay datos de modelo disponibles"}

@layervisrouter.get("/model_version_layer_visualizer")
async def get_model_version():
    return {"version": model_version}
