import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import cv2

VALID_IMAGE_NAMES = ['astronaut', 'camera', 'coins', 'horse', 'moon', 'page', 'clock', 'text', 'coffee', 'rocket', 'chelsea', 'cat', 'horse']
VALID_MODEL_NAMES = ['vgg16', 'alexnet', 'resnet50']

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

def get_sample_image(name):
    if name == 'astronaut':
        return data.astronaut()
    elif name == 'camera':
        return data.camera()
    elif name == 'coins':
        return data.coins()
    elif name == 'horse':
        return data.horse()
    elif name == 'moon':
        return data.moon()
    elif name == 'page':
        return data.page()
    elif name == 'clock':
        return data.clock()
    elif name == 'text':
        return data.text()
    elif name=='coffee':
        return data.coffee()
    elif name=='rocket':
        return data.rocket()
    elif name=='chelsea':
        return data.chelsea()
    elif name=='cat':
        return data.cat()
    elif name=='horse':
        return data.horse()
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

def get_activation_map(input_image, model, layer_idx):
    # Obtener la capa de la que queremos obtener los mapas de activación
    conv_layer = model.features[layer_idx]
    with torch.no_grad():
        activation_map = conv_layer(input_image)

    # Convertir los mapas de activación a numpy
    activation_map = activation_map.squeeze().numpy()

    # Calcular el promedio a través de todos los mapas de activación
    activation_map_combined = np.mean(activation_map, axis=0)

    # Normalizar el mapa combinado para que esté entre 0 y 1
    activation_map_combined -= activation_map_combined.min()
    activation_map_combined /= activation_map_combined.max()
    return activation_map_combined
