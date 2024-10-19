const networkContainer = document.getElementById('networkContainer');
const layerConfigDiv = document.getElementById('layerConfig');
const updateNetworkButton = document.getElementById('updateNetwork');

// Inicializar la configuración de la red
let layers = [
    { neurons: 3 },  // Capa de entrada
    { neurons: 4, activations: [], biases: [] },  // Capa oculta 1
    { neurons: 4, activations: [], biases: [] },  // Capa oculta 2
    { neurons: 2, activations: [], biases: [], outputs: [] }   // Capa de salida
];

let neurons = [];
let connections = [];
let biases = []; // Almacenar los valores de sesgo para cada capa

function initLayerConfig() {
    layerConfigDiv.innerHTML = '';
    layers.forEach((layer, index) => {
        const layerDiv = document.createElement('div');
        layerDiv.classList.add('layer-config');
        const label = document.createElement('label');
        label.innerText = `Capa ${index + 1} - Neuronas: `;
        const neuronInput = document.createElement('input');
        neuronInput.type = 'number';
        neuronInput.min = '1';
        neuronInput.value = layer.neurons;
        neuronInput.dataset.layerIndex = index;
        neuronInput.addEventListener('change', updateLayerNeurons);

        label.appendChild(neuronInput);
        layerDiv.appendChild(label);

        // Agregar selección de función de activación (excepto para la capa de entrada)
        if (index > 0) {
            const activationSelect = document.createElement('select');
            activationSelect.classList.add('activation-select');
            activationSelect.dataset.layerIndex = index;
            const activations = ['σ', 'ReLU', 'tanh', 'none'];
            // Añadir 'softmax' solo para la última capa
            if (index === layers.length - 1) {
                activations.push('softmax');
            }
            activations.forEach(func => {
                const option = document.createElement('option');
                option.value = func;
                option.text = func === 'none' ? '-' : func;
                activationSelect.appendChild(option);
            });
            activationSelect.value = layer.activations[0] || 'σ';
            activationSelect.addEventListener('change', updateActivationFunction);
            layerDiv.appendChild(activationSelect);
        }

        layerConfigDiv.appendChild(layerDiv);
    });
}

function updateLayerNeurons(event) {
    const layerIndex = event.target.dataset.layerIndex;
    const neuronCount = parseInt(event.target.value);
    layers[layerIndex].neurons = neuronCount;
    // Reiniciar activations, biases y outputs si existen
    if (layerIndex > 0) {
        layers[layerIndex].activations = new Array(neuronCount).fill(layers[layerIndex].activations[0] || 'σ');
        layers[layerIndex].biases = [];
    }
    if (layerIndex == layers.length - 1) {
        layers[layerIndex].outputs = [];
    }
}

function updateActivationFunction(event) {
    const layerIndex = event.target.dataset.layerIndex;
    const activationFunc = event.target.value;
    const neuronCount = layers[layerIndex].neurons;
    layers[layerIndex].activations = new Array(neuronCount).fill(activationFunc);
}

updateNetworkButton.addEventListener('click', initNetwork);

function initNetwork() {
    neurons = [];
    connections = [];
    biases = [];
    networkContainer.innerHTML = '';

    const width = networkContainer.clientWidth;
    const height = networkContainer.clientHeight;

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
                let activationFunc = layers[layerIndex].activations[i] || 'σ';
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
        inputBox.addEventListener('input', updateOutputs);
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
        biasSlider.value = '0';
        biasSlider.addEventListener('input', () => {
            biasNeuron.innerText = parseFloat(biasSlider.value).toFixed(1);
            updateOutputs();
        });

        biasNeuron.innerText = '0.0';

        biasSliderContainer.appendChild(biasSlider);
        networkContainer.appendChild(biasNeuron);
        networkContainer.appendChild(biasSliderContainer); // Colocar el slider después del biasNeuron para que esté encima

        biases[layerIndex - 1] = biasSlider;
    }

    // Dibujar sliders sin líneas de conexión
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
                slider.min = '-1';
                slider.max = '10';
                slider.step = '0.01';
                slider.value = '0';
                slider.classList.add('slider');
                slider.style.width = `${length}px`;
                slider.style.left = `${startX}px`;
                slider.style.top = `${startY - 2}px`; // Ajuste para altura del slider
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

    updateOutputs();
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
    line.style.transform = `rotate(${angle}deg)`;
    networkContainer.appendChild(line);
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

        const biasValue = parseFloat(biases[layerIndex - 1].value) || 0;

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

// Inicializar la configuración y la red al cargar la página
initLayerConfig();
initNetwork();
