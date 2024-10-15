const neuronContainer = document.getElementById('neuronContainer');
const updateNeuronButton = document.getElementById('updateNeuron');
const numInputsInput = document.getElementById('numInputs');
const activationSelect = document.getElementById('activationSelect');

updateNeuronButton.addEventListener('click', initNeuron);

function initNeuron() {
    const numInputs = parseInt(numInputsInput.value);
    neuronContainer.innerHTML = '';

    const width = neuronContainer.clientWidth;
    const height = neuronContainer.clientHeight;

    const inputSpacing = height / (numInputs + 1);
    const startX = 50;

    const inputs = [];
    const weights = [];

    const circleDiameter = 50;

    // Coordenadas centrales
    const sumCircleCenterX = width / 2;
    const sumCircleCenterY = height / 2;

    const activationCircleCenterX = width / 2 + 150;
    const activationCircleCenterY = height / 2;

    // Crear inputs y sliders
    for (let i = 0; i < numInputs; i++) {
        const y = inputSpacing * (i + 1);

        // Caja de input
        const inputBox = document.createElement('input');
        inputBox.type = 'number';
        inputBox.min = '-100';
        inputBox.max = '100';
        inputBox.step = '0.1';
        inputBox.value = '0';
        inputBox.classList.add('input-box');
        inputBox.style.left = `${startX - 60}px`;
        inputBox.style.top = `${y - 15}px`;
        inputBox.addEventListener('input', updateOutput);
        neuronContainer.appendChild(inputBox);
        inputs.push(inputBox);

        // Línea desde inputBox hasta slider (más corta)
        drawLine(startX, y, startX + 50, y);

        // Slider para el peso (más largo)
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '-10';
        slider.max = '10';
        slider.step = '0.1';
        slider.value = '1';
        slider.classList.add('slider');
        slider.style.width = '150px'; // Slider más largo
        slider.style.left = `${startX + 50}px`;
        slider.style.top = `${y - 2}px`;
        slider.addEventListener('input', updateOutput);
        neuronContainer.appendChild(slider);
        weights.push(slider);

        // Línea desde slider hasta sumatorio
        const sliderEndX = startX + 200;
        const sliderEndY = y;
        drawLine(sliderEndX, sliderEndY, sumCircleCenterX - circleDiameter / 2, sumCircleCenterY);
    }

    // Caja para el sesgo (bias)
    const biasBox = document.createElement('input');
    biasBox.type = 'number';
    biasBox.min = '-100';
    biasBox.max = '100';
    biasBox.step = '0.1';
    biasBox.value = '0';
    biasBox.classList.add('bias-box');
    biasBox.style.left = `${sumCircleCenterX - 30}px`;
    biasBox.style.top = `${sumCircleCenterY - 120}px`;
    biasBox.addEventListener('input', updateOutput);
    neuronContainer.appendChild(biasBox);

    // Línea desde biasBox hasta sumatorio
    drawLine(sumCircleCenterX, sumCircleCenterY - circleDiameter / 2, sumCircleCenterX, sumCircleCenterY - 90);

    // Círculo del sumatorio (Sigma) centrado
    const sumCircle = document.createElement('div');
    sumCircle.classList.add('neuron-circle');
    sumCircle.innerText = '∑';
    sumCircle.style.left = `${sumCircleCenterX - circleDiameter / 2}px`;
    sumCircle.style.top = `${sumCircleCenterY - circleDiameter / 2}px`;
    neuronContainer.appendChild(sumCircle);

    // Caja para mostrar la preactivación debajo de Sigma
    const preActivationBox = document.createElement('div');
    preActivationBox.classList.add('output-box');
    preActivationBox.style.left = `${sumCircleCenterX - 40}px`;
    preActivationBox.style.top = `${sumCircleCenterY + circleDiameter / 2 + 10}px`;
    neuronContainer.appendChild(preActivationBox);

    // Línea vertical desde sumatorio a preactivación
    drawLine(sumCircleCenterX, sumCircleCenterY + circleDiameter / 2, sumCircleCenterX, sumCircleCenterY + circleDiameter / 2 + 10);

    // Línea desde sumatorio hasta función de activación
    drawLine(sumCircleCenterX + circleDiameter / 2, sumCircleCenterY, activationCircleCenterX - circleDiameter / 2, activationCircleCenterY);

    // Círculo de la función de activación centrado
    const activationCircle = document.createElement('div');
    activationCircle.classList.add('activation-circle');
    activationCircle.innerText = 'f';
    activationCircle.style.left = `${activationCircleCenterX - circleDiameter / 2}px`;
    activationCircle.style.top = `${activationCircleCenterY - circleDiameter / 2}px`;
    neuronContainer.appendChild(activationCircle);

    // Línea desde función de activación hasta output
    const outputBoxX = width - 90;
    const outputBoxY = activationCircleCenterY - 15;

    drawLine(activationCircleCenterX + circleDiameter / 2, activationCircleCenterY, outputBoxX, activationCircleCenterY);

    // Caja de output
    const outputBox = document.createElement('div');
    outputBox.classList.add('output-box');
    outputBox.style.left = `${outputBoxX}px`;
    outputBox.style.top = `${outputBoxY}px`;
    neuronContainer.appendChild(outputBox);

    // Actualizar output inicialmente
    updateOutput();

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
        neuronContainer.appendChild(line);
    }

    function updateOutput() {
        let preActivation = 0;
        for (let i = 0; i < numInputs; i++) {
            const inputValue = parseFloat(inputs[i].value) || 0;
            const weightValue = parseFloat(weights[i].value) || 0;
            preActivation += inputValue * weightValue;
        }
        // Añadir sesgo
        const biasValue = parseFloat(biasBox.value) || 0;
        preActivation += biasValue;

        preActivationBox.innerText = preActivation.toFixed(2);

        const activationFunc = activationSelect.value;
        const outputValue = applyActivationFunction(preActivation, activationFunc);
        outputBox.innerText = outputValue.toFixed(2);
    }

    function applyActivationFunction(x, funcName) {
        switch (funcName) {
            case 'σ':
                return 1 / (1 + Math.exp(-x));
            case 'ReLU':
                return Math.max(0, x);
            case 'tanh':
                return Math.tanh(x);
            case 'none':
            case '-':
            default:
                return x;
        }
    }
}

// Inicializar la neurona al cargar la página
initNeuron();
