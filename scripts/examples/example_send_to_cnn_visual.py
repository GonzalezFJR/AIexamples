import numpy as np
from model_layers_visualizer import send_to_cnn_visualizer

if __name__ == '__main__':
    N = 10
    matrix = np.random.randint(0, 256, size=(N, N)).tolist()
    data = {'matrix': matrix}
    send_to_cnn_visualizer(data, 'localhost', 8000)

