import torch.nn as nn

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

    data = {
        'layers': layers
    }
    return data
