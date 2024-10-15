// script.js

const networkContainer = document.getElementById('networkContainer');

// Inicializar la configuración de la red
let layers = []; // Será llenado con datos del servidor
let neurons = [];
let connections = [];
let biases = []; // Almacenar los valores de sesgo para cada capa

// Función para inicializar la conexión WebSocket
function initWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/dnnvis/ws');

    ws.onopen = function () {
        console.log('WebSocket connection opened');
    };

    ws.onmessage = function (event) {
        const data = JSON.parse(event.data);
        if (data.type === 'network_state') {
            layers = data.layers;
            const inputs = data.inputs;
            initNetwork();
            updateInputs(inputs);
        }
    };

    ws.onclose = function () {
        console.log('WebSocket connection closed, retrying in 1 second...');
        setTimeout(initWebSocket, 1000); // Reintentar conexión en 1 segundo
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
        ws.close();
    };
}

// Inicializar la red neuronal
function initNetwork() {
    neurons = [];
    connections = [];
    biases = [];
    networkContainer.innerHTML = '';

    const width = networkContainer.clientWidth || 800;
    const height = networkContainer.clientHeight || 600;

    const layerSpacing = width / (layers.length + 1);
    const neuronRadius = 20;

    // Posicionar neuronas
    layers.forEach((layer, layerIndex) => {
        const x = layerSpacing * (layerIndex + 1);
        const neuronCount = layer.neurons;
        const ySpacing = height / (neuronCount + 1);

        neurons[layerIndex] = [];

        for (let i = 0; i < neuronCount; i++) {
            const y = ySpacing * (i + 1);
            const neuron = document.createElement('div');
            neuron.classList.add('neuron');
            neuron.style.left = `${x - neuronRadius}px`;
            neuron.style.top = `${y - neuronRadius}px`;

            if (layerIndex > 0) {
                // Mostrar función de activación
                let activationFunc = layer.activations[i] || 'σ';
                if (activationFunc === 'none') {
                    activationFunc = '-';
                }
                neuron.innerText = activationFunc;
                if (activationFunc === 'ReLU' || activationFunc === 'tanh' || activationFunc === 'softmax') {
                    neuron.style.fontSize = '12px';
                } else {
                    neuron.style.fontSize = '16px';
                }
            }

            networkContainer.appendChild(neuron);
            neurons[layerIndex].push({ element: neuron, x: x, y: y });
        }
    });

    // Añadir cajas de entrada
    const inputLayerIndex = 0;
    const inputLayerNeurons = neurons[inputLayerIndex];
    layers[inputLayerIndex].inputs = [];

    inputLayerNeurons.forEach((neuron, index) => {
        const inputBox = document.createElement('input');
        inputBox.type = 'number';
        inputBox.min = '-100';
        inputBox.max = '100';
        inputBox.step = '0.01';
        inputBox.value = '0';
        inputBox.classList.add('input-box');
        inputBox.style.left = `${neuron.x - 120}px`; // Ajuste para mayor tamaño
        inputBox.style.top = `${neuron.y - 15}px`;
        inputBox.disabled = true; // Deshabilitar edición manual
        networkContainer.appendChild(inputBox);
        layers[inputLayerIndex].inputs.push(inputBox);

        // Línea desde la caja de entrada a la neurona
        drawLine(inputBox.offsetLeft + 80, neuron.y, neuron.x - neuronRadius, neuron.y);
    });

    // Añadir bias neurons y sliders
    for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
        const x = (neurons[layerIndex - 1][0].x + neurons[layerIndex][0].x) / 2;
        const y = networkContainer.clientHeight - 100; // Ajuste de posición

        const biasNeuron = document.createElement('div');
        biasNeuron.classList.add('bias-neuron');
        biasNeuron.style.left = `${x - 25}px`;
        biasNeuron.style.top = `${y}px`;

        const biasSliderContainer = document.createElement('div');
        biasSliderContainer.classList.add('circular-slider');
        biasSliderContainer.style.left = `${x - 25}px`;
        biasSliderContainer.style.top = `${y}px`;

        const biasSlider = document.createElement('input');
        biasSlider.type = 'range';
        biasSlider.min = '-100';
        biasSlider.max = '100';
        biasSlider.step = '0.1';

        // Establecer el valor del slider desde los datos
        const biasValue = layers[layerIndex].biases[0] || 0;
        biasSlider.value = biasValue;
        biasNeuron.innerText = parseFloat(biasValue).toFixed(1);

        biasSlider.addEventListener('input', () => {
            biasNeuron.innerText = parseFloat(biasSlider.value).toFixed(1);
            biases[layerIndex - 1] = biasSlider.value;
            updateOutputs();
        });

        biasSliderContainer.appendChild(biasSlider);
        networkContainer.appendChild(biasNeuron);
        networkContainer.appendChild(biasSliderContainer); // Colocar el slider después del biasNeuron para que esté encima

        biases[layerIndex - 1] = biasSlider.value;
    }

    // Añadir sliders para pesos y conexiones
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
        const currentLayerNeurons = neurons[layerIndex];
        const nextLayerNeurons = neurons[layerIndex + 1];

        connections[layerIndex] = [];

        for (let i = 0; i < currentLayerNeurons.length; i++) {
            for (let j = 0; j < nextLayerNeurons.length; j++) {
                const startX = currentLayerNeurons[i].x + neuronRadius;
                const startY = currentLayerNeurons[i].y;
                const endX = nextLayerNeurons[j].x - neuronRadius;
                const endY = nextLayerNeurons[j].y;

                // Calcular posición y rotación del slider
                const deltaX = endX - startX;
                const deltaY = endY - startY;
                const length = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
                const angle = Math.atan2(deltaY, deltaX) * 180 / Math.PI;

                // Añadir slider como conexión
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '-10';
                slider.max = '10';
                slider.step = '0.01';

                // Establecer el valor del slider desde los datos
                const weight = layers[layerIndex + 1].weights[j][i] || 0;
                slider.value = weight;

                slider.classList.add('slider');
                slider.style.width = `${length}px`;
                slider.style.left = `${startX}px`;
                slider.style.top = `${startY - 2}px`; // Ajuste para altura del slider
                slider.style.transformOrigin = '0% 50%';
                slider.style.transform = `rotate(${angle}deg)`;
                slider.addEventListener('input', updateOutputs);
                networkContainer.appendChild(slider);

                connections[layerIndex].push({
                    from: i,
                    to: j,
                    weight: slider
                });
            }
        }
    }

    // Añadir cajas de salida
    const outputLayerIndex = layers.length - 1;
    const outputLayerNeurons = neurons[outputLayerIndex];
    layers[outputLayerIndex].outputs = [];

    outputLayerNeurons.forEach((neuron, index) => {
        const outputBox = document.createElement('div');
        outputBox.classList.add('output-box');
        outputBox.style.left = `${neuron.x + 50}px`;
        outputBox.style.top = `${neuron.y - 15}px`;
        networkContainer.appendChild(outputBox);
        layers[outputLayerIndex].outputs.push(outputBox);

        // Línea desde la neurona a la caja de salida
        drawLine(neuron.x + neuronRadius, neuron.y, outputBox.offsetLeft, neuron.y);
    });
}

// Actualizar los inputs desde los datos obtenidos del servidor
function updateInputs(inputs) {
    const inputLayer = layers[0];
    inputLayer.inputs.forEach((inputBox, index) => {
        inputBox.value = inputs[index] || 0;
    });
    updateOutputs();
}

// Actualizar los valores de salida
function updateOutputs() {
    // Obtener valores de entrada
    const inputLayer = layers[0];
    const inputs = inputLayer.inputs.map(input => parseFloat(input.value) || 0);

    let activations = [];
    activations[0] = inputs;

    // Propagación hacia adelante
    for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
        const previousActivations = activations[layerIndex - 1];
        const currentLayer = layers[layerIndex];
        let layerActivations = [];

        const biasValue = parseFloat(biases[layerIndex - 1]) || 0;

        let sums = []; // Para softmax

        for (let n = 0; n < currentLayer.neurons; n++) {
            let sum = 0;
            for (let p = 0; p < previousActivations.length; p++) {
                const weight = getWeight(layerIndex - 1, p, n);
                sum += previousActivations[p] * weight;
            }
            // Añadir sesgo
            sum += biasValue;
            sums.push(sum);
        }

        const activationFunc = currentLayer.activations[0] || 'σ';

        if (activationFunc === 'softmax') {
            layerActivations = softmax(sums);
        } else {
            for (let sum of sums) {
                const activation = applyActivationFunction(sum, activationFunc);
                layerActivations.push(activation);
            }
        }

        activations[layerIndex] = layerActivations;
    }

    // Actualizar cajas de salida
    const outputLayer = layers[layers.length - 1];
    outputLayer.outputs.forEach((outputBox, index) => {
        outputBox.innerText = activations[layers.length - 1][index].toFixed(2);
    });
}

function getWeight(layerIndex, fromNeuron, toNeuron) {
    const connectionList = connections[layerIndex];
    for (let conn of connectionList) {
        if (conn.from === fromNeuron && conn.to === toNeuron) {
            return parseFloat(conn.weight.value) || 0;
        }
    }
    return 0;
}

function applyActivationFunction(x, funcName, previousActivations = []) {
    switch (funcName) {
        case 'σ':
            return 1 / (1 + Math.exp(-x));
        case 'ReLU':
            return Math.max(0, x);
        case 'tanh':
            return Math.tanh(x);
        case 'softmax':
            // La función softmax se aplicará después de calcular todos los x
            return x; // Retornamos x para procesarlo luego
        case 'none':
        case '-':
        default:
            return x;
    }
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(exp => exp / sum);
}

function drawLine(x1, y1, x2, y2) {
    const deltaX = x2 - x1;
    const deltaY = y2 - y1;
    const length = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const angle = Math.atan2(deltaY, deltaX) * 180 / Math.PI;

    const line = document.createElement('div');
    line.classList.add('connection');
    line.style.width = `${length}px`;
    line.style.left = `${x1}px`;
    line.style.top = `${y1}px`;
    line.style.transformOrigin = '0% 50%';
    line.style.transform = `rotate(${angle}deg)`;
    networkContainer.appendChild(line);
}

// Iniciar la conexión WebSocket al cargar la página
initWebSocket();
