document.addEventListener('DOMContentLoaded', function() {
    let currentVersion = 0;

    function checkForUpdates() {
        fetch('/model_version_layer_visualizer')
            .then(response => response.json())
            .then(data => {
                if (data.version !== currentVersion) {
                    currentVersion = data.version;
                    // Recargar la visualización
                    fetchModelAndVisualize();
                }
            })
            .catch(error => console.error('Error al obtener la versión del modelo:', error));
    }

    function fetchModelAndVisualize() {
        fetch('/get_model_layer_visualizer')
            .then(response => response.json())
            .then(data => {
                // Limpiar el contenedor
                const container = document.getElementById('visualization-container');
                container.innerHTML = '';
                // Visualizar el nuevo modelo
                visualizeCNN(data);
            })
            .catch(error => console.error('Error al cargar el JSON:', error));
    }

    function visualizeCNN(layers) {
        const container = document.getElementById('visualization-container');

        // Capa de entrada
        const inputLayer = document.createElement('div');
        inputLayer.className = 'layer-box input-layer';
        inputLayer.innerHTML = `Entrada<br>Tamaño: ${layers[0].input_shape.slice(1).join(' x ')}`;
        container.appendChild(inputLayer);

        layers.forEach((layer, index) => {
            // Flecha
            const arrow = document.createElement('div');
            arrow.className = 'arrow';
            container.appendChild(arrow);

            const layerBox = document.createElement('div');
            layerBox.className = 'layer-box';

            switch (layer.type) {
                case 'Conv2d':
                    layerBox.classList.add('conv-layer');
                    layerBox.innerHTML = `<strong>Convolución</strong><br>
                                          Kernel: ${layer.kernel_size}<br>
                                          Stride: ${layer.stride}<br>
                                          Padding: ${layer.padding}<br>
                                          Número de filtros: ${layer.num_filters}<br>
                                          Salida: ${layer.output_shape.slice(1).join(' x ')}`;
                    break;
                case 'MaxPool2d':
                case 'AvgPool2d':
                case 'AdaptiveAvgPool2d':
                    layerBox.classList.add('pool-layer');
                    layerBox.innerHTML = `<strong>${layer.type}</strong><br>
                                          Kernel: ${layer.kernel_size || ''}<br>
                                          Stride: ${layer.stride || ''}<br>
                                          Padding: ${layer.padding || ''}<br>
                                          Salida: ${layer.output_shape.slice(1).join(' x ')}`;
                    break;
                case 'ReLU':
                case 'Sigmoid':
                case 'Tanh':
                    layerBox.classList.add('activation-layer');
                    layerBox.innerHTML = `<strong>${layer.type}</strong>`;
                    break;
                case 'Dropout':
                    layerBox.classList.add('dropout-layer');
                    layerBox.innerHTML = `<strong>Dropout</strong>`;
                    break;
                case 'Linear':
                    layerBox.classList.add('fc-layer');
                    layerBox.innerHTML = `<strong>Capa Fully Connected</strong><br>
                                          Entradas: ${layer.in_features}<br>
                                          Salida: ${layer.out_features}`;
                    break;
                case 'Flatten':
                    layerBox.classList.add('flatten-layer');
                    layerBox.innerHTML = `<strong>Flatten</strong>`;
                    break;
                case 'LSTM':
                case 'GRU':
                case 'RNN':
                    layerBox.classList.add('rnn-layer');
                    layerBox.innerHTML = `<strong>${layer.type}</strong><br>
                                          Input Size: ${layer.input_size}<br>
                                          Hidden Size: ${layer.hidden_size}<br>
                                          Num Layers: ${layer.num_layers}<br>
                                          Bidireccional: ${layer.bidirectional}<br>
                                          Salida: ${layer.output_shape.slice(1).join(' x ')}`;
                    break;
                default:
                    layerBox.classList.add('other-layer');
                    layerBox.innerHTML = `<strong>${layer.type}</strong>`;
            }
            container.appendChild(layerBox);
        });

        // Flecha final
        const finalArrow = document.createElement('div');
        finalArrow.className = 'arrow';
        container.appendChild(finalArrow);

        // Capa de salida
        const outputLayer = document.createElement('div');
        outputLayer.className = 'layer-box output-layer';
        const lastLayer = layers[layers.length - 1];
        outputLayer.innerHTML = `Salida<br>Tamaño: ${lastLayer.output_shape.slice(1).join(' x ')}`;
        container.appendChild(outputLayer);
    }

    // Obtener y visualizar el modelo inicialmente
    fetchModelAndVisualize();

    // Configurar la verificación periódica cada 5 segundos
    setInterval(checkForUpdates, 5000);
});
