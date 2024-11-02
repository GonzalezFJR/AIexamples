import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import cv2
import base64
import requests

VALID_IMAGE_NAMES = ['astronaut', 'coffee', 'rocket', 'cat']
VALID_MODEL_NAMES = ['vgg16', 'alexnet', 'resnet50']

# Diccionario global para almacenar modelos e imágenes personalizados
custom_models = {}


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando el dispositivo {device}')
    return device

def get_model(name=None):
    if name is None: name = 'vgg16'
    if name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f'Modelo no reconocido: {name}')
    model.eval()
    device = get_device()
    model.to(device)
    return model

def load_and_prepar_img(imgname, resize=(224, 224)):
    image = cv2.imread(imgname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (resize[1], resize[0]))
    image = image / 255.0
    image = image.transpose((2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image, image_tensor  

def get_model_layers(model):
    layers = []
    idx = 0

    def traverse_model(module, parent_name=''):
        nonlocal idx
        for name, submodule in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(submodule, torch.nn.Conv2d):
                layers.append({"idx": idx, "name": full_name, "type": str(submodule)})
                idx += 1
            else:
                traverse_model(submodule, full_name)

    traverse_model(model)
    return layers

def print_model_layers(model):
    layers = get_model_layers(model)
    for layer in layers:
        print(f"Índice: {layer['idx']}, Nombre: {layer['name']}, Tipo: {layer['type']}")

def get_activation_map_layer(input_image, model, layer_idx):
    activation_map = None
    activations = {}
    idx = 0

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    handles = []

    def register_hooks(module):
        nonlocal idx
        for name, submodule in module.named_children():
            if isinstance(submodule, torch.nn.Conv2d):
                if idx == layer_idx:
                    handle = submodule.register_forward_hook(get_activation(name))
                    handles.append(handle)
                idx += 1
            else:
                register_hooks(submodule)

    register_hooks(model)

    # Pasar la imagen a través del modelo
    with torch.no_grad():
        model(input_image)

    # Eliminar los hooks
    for handle in handles:
        handle.remove()

    # Obtener la activación
    if activations:
        activation_map = list(activations.values())[0]
    else:
        raise ValueError("No se encontró la capa especificada")

    # Procesar el mapa de activación
    activation_map = activation_map.squeeze().cpu().numpy()
    activation_map_combined = np.mean(activation_map, axis=0)

    # Normalizar
    activation_map_combined -= activation_map_combined.min()
    activation_map_combined /= activation_map_combined.max()
    return activation_map_combined

def send_activation_map(image, activation_map, host='localhost', port=8000):
    server_url=f"http://{host}:{port}/activation_maps/custom_activation_map"):
    # Convertir las imágenes a formato base64
    # image is of shape (3, 224, 224) --> go to (224, 224, 3)
    image = image.transpose((1, 2, 0))
    _, image_buffer = cv2.imencode('.png', np.uint8(255 * image))
    _, activation_map_buffer = cv2.imencode('.jpg', cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET))
    image_base64 = base64.b64encode(image_buffer).decode('utf-8')
    activation_map_base64 = base64.b64encode(activation_map_buffer).decode('utf-8')

    data = {
        'image': image_base64,
        'activation_map': activation_map_base64
    }

    response = requests.post(server_url, json=data)

    if response.status_code == 200:
        visualization_url = f"http://{host}:{port}/activation_maps/custom_visualization/"
        return visualization_url
    else:
        print(f'Error: {response.text}')
        return None


def get_sample_image(name):
    if name == 'astronaut':
        return data.astronaut()
    elif name=='coffee':
        return data.coffee()
    elif name=='rocket':
        return data.rocket()
    elif name=='cat':
        return data.cat()
    return None

def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)
    return input_image


def get_activation_map(input_image, model, layer_identifier, model_name):
    # Obtener el dispositivo del modelo
    device = next(model.parameters()).device

    # Asegurarse de que la imagen está en el dispositivo correcto
    input_image = input_image.to(device)

    activation_map = None
    activations = {}
    idx = 0  # Contador de índices de capa

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Función recursiva para registrar hooks en la capa seleccionada
    def register_hooks(module, parent_name=''):
        nonlocal idx
        handles = []
        for name, submodule in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if hasattr(submodule, "original_name"):
                if submodule.original_name == "Conv2d":
                    if idx == layer_identifier:
                        handle = submodule.register_forward_hook(get_activation(full_name))
                        handles.append(handle)
                    idx += 1
            elif isinstance(submodule, torch.nn.Conv2d):
                if idx == layer_identifier:
                    handle = submodule.register_forward_hook(get_activation(full_name))
                    handles.append(handle)
                idx += 1
            else:
                handles.extend(register_hooks(submodule, full_name))
        return handles

    # Registrar hooks en las capas del modelo
    handles = register_hooks(model)

    # Pasar la imagen a través del modelo completo
    with torch.no_grad():
        model(input_image)

    # Eliminar los hooks
    for handle in handles:
        handle.remove()

    # Obtener la activación de la capa seleccionada
    if activations:
        activation_map = list(activations.values())[0]
    else:
        raise ValueError("No se encontró la capa especificada en el modelo")

    # Convertir el mapa de activación a CPU y Numpy
    activation_map = activation_map.squeeze().cpu().numpy()

    # Calcular el promedio a través de los canales
    activation_map_combined = np.mean(activation_map, axis=0)

    # Normalizar el mapa combinado
    activation_map_combined -= activation_map_combined.min()
    activation_map_combined /= activation_map_combined.max()

    return activation_map_combined

def visualize_activation_map(imgname, model, layer, host='localhost', port=8000):
    # Cargar una imagen con cv2
    print('Capas del modelo:')
    print_model_layers(model)
    print("...cargando visualización de capa", layer)
    image, image_tensor = load_and_prepar_img(imgname)
    activation_map = get_activation_map_layer(image_tensor, model, layer)
    visualization_url = send_activation_map(image, activation_map, host, port)
    print("Visualización generada:", visualization_url)
    return visualization_url


############################################################################################
############################################################################################
############################################################################################

from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import uuid
import io
from PIL import Image
import base64

actrouter = APIRouter()

FIXED_MODEL_NAME = None

# Ruta para obtener la lista de imágenes de ejemplo
@actrouter.get("/images")
def get_images():
    return {"images": VALID_IMAGE_NAMES}

# Ruta para obtener la lista de modelos
@actrouter.get("/models")
def get_models():
    if FIXED_MODEL_NAME:
        return {"models": [FIXED_MODEL_NAME]}
    else:
        return {"models": VALID_MODEL_NAMES}


@actrouter.get("/layers/{model_name}")
def get_layers(model_name: str):
    if model_name == 'custom':
        keys = list(custom_models.keys())
        if len(keys) >= 1:
            custom_id = keys[-1]
        else:
            return JSONResponse(status_code=400, content={"error": "No hay ningún modelo cargado!"})
        model = custom_models[custom_id]['model']
    else:
        try:
            model = get_model(model_name)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Modelo no reconocido"})
    
    layers = []
    idx = 0  # Contador de índices de capa

    # Función recursiva para recorrer el modelo y sus submódulos
    def traverse_model(module, parent_name=''):
        nonlocal idx
        for name, submodule in module.named_children():
            print('name = ', name)
            print('submodule = ', submodule)
            full_name = f"{parent_name}.{name}" if parent_name else name
            if hasattr(submodule, "original_name"):
                if submodule.original_name == "Conv2d":
                    layers.append({"idx": idx, "name": full_name, "type": str(submodule)})
                    idx += 1
            elif isinstance(submodule, torch.nn.Conv2d):
                layers.append({"idx": idx, "name": full_name, "type": str(submodule)})
                idx += 1
            else:
                traverse_model(submodule, full_name)

    # Llamar a la función recursiva sobre el modelo
    traverse_model(model)
    print('model = ', model)

    return {"layers": layers}

@actrouter.post("/activation_map")
async def activation_map(request: Request,
                         image_name: str = Form(None),
                         image_file: UploadFile = File(None),
                         model_name: str = Form(None),
                         layer_idx: int = Form(...),
                         custom_id: str = Form(None)):
    # Manejar el modelo
    if model_name == 'custom':
        keys = list(custom_models.keys())
        if len(keys) >= 1:
            custom_id = keys[-1]
        else:
            return JSONResponse(status_code=400, content={"error": "No hay ningún modelo cargado!"})
        model = custom_models[custom_id]['model']
        image = custom_models[custom_id]['image']
    else:
        if FIXED_MODEL_NAME:
            model_name = FIXED_MODEL_NAME
        try:
            model = get_model(model_name)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Modelo no reconocido"})
        # Manejar la imagen
        if image_name:
            image = get_sample_image(image_name)
            if image is None:
                return JSONResponse(status_code=400, content={"error": "Imagen no válida"})
        elif image_file:
            contents = await image_file.read()
            image = np.array(Image.open(io.BytesIO(contents)))
        else:
            return JSONResponse(status_code=400, content={"error": "Se requiere una imagen"})
    
    # Transformar la imagen
    input_image = transform_image(image)
    # Obtener el mapa de activación
    activation = get_activation_map(input_image, model, layer_idx, model_name)
    # Redimensionar el mapa de activación al tamaño de la imagen
    activation_resized = cv2.resize(activation, (image.shape[1], image.shape[0]))
    # Convertir el mapa de activación a color
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_resized), cv2.COLORMAP_JET)
    # Convertir imágenes a bytes
    _, image_buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    _, heatmap_buffer = cv2.imencode('.png', heatmap)
    image_base64 = base64.b64encode(image_buffer).decode('utf-8')
    heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')
    return {"original_image": image_base64, "activation_map": heatmap_base64}


@actrouter.post("/upload_custom_model")
async def upload_custom_model(model_file: UploadFile = File(...), image_file: UploadFile = File(...)):
    # Generar un identificador único
    custom_id = str(uuid.uuid4())
    
    # Cargar el modelo
    contents = await model_file.read()
    
    # Por razones de seguridad, recomendamos usar TorchScript para serializar el modelo
    buffer = io.BytesIO(contents)
    model = torch.jit.load(buffer)
    model.eval()
    device = get_device()
    model.to(device)
    
    # Cargar la imagen
    image_contents = await image_file.read()
    image = np.array(Image.open(io.BytesIO(image_contents)))
    
    # Almacenar el modelo y la imagen en el diccionario global
    custom_models[custom_id] = {'model': model, 'image': image}
    
    # Devolver el custom_id al cliente
    return {'custom_id': custom_id}

from fastapi import FastAPI, Body, HTTPException
custom_data = {}

@actrouter.post("/custom_activation_map")
async def custom_activation_map(data: dict = Body(...)):
    image_base64 = data.get('image')
    activation_map_base64 = data.get('activation_map')
    if not image_base64 or not activation_map_base64:
        raise HTTPException(status_code=400, detail="Se requieren la imagen y el mapa de activación")

    # Generar un identificador único
    custom_id = str(uuid.uuid4())

    # Almacenar los datos
    global custom_data
    custom_data = {
        'original_image': image_base64,
        'activation_map': activation_map_base64
    }
    return {'status': 'success'}

from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@actrouter.get("/custom_visualization/")
async def custom_visualization(request: Request):
    return templates.TemplateResponse("activation_map_visualization.html", {"request": request, "data": custom_data})

