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
