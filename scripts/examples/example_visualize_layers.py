from torchvision import models
from model_layers_visualizer import visualize

if __name__ == '__main__':
    model = models.alexnet()
    input_shape = (1, 3, 224, 224)
    visualize(model, input_shape, 'localhost', 8000)
