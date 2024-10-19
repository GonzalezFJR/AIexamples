document.getElementById('updateButton').addEventListener('click', drawNetwork);

function drawNetwork() {
    const inputNeurons = parseInt(document.getElementById('inputNeurons').value);
    const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
    const neuronsPerLayer = parseInt(document.getElementById('neuronsPerLayer').value);
    const outputNeurons = parseInt(document.getElementById('outputNeurons').value);

    const layers = [];

    layers.push(inputNeurons);
    for (let i = 0; i < hiddenLayers; i++) {
        layers.push(neuronsPerLayer);
    }
    layers.push(outputNeurons);

    const canvas = document.getElementById('networkCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layerSpacing = canvas.width / (layers.length + 1);

    const neuronRadius = 15;

    const positions = [];

    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const x = (i + 1) * layerSpacing;
        const ySpacing = canvas.height / (layer + 1);
        positions[i] = [];
        for (let j = 0; j < layer; j++) {
            const y = (j + 1) * ySpacing;
            positions[i][j] = { x: x, y: y };
            ctx.beginPath();
            ctx.arc(x, y, neuronRadius, 0, 2 * Math.PI);
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = '#000';
            ctx.fill();
            ctx.stroke();
        }
    }

    // Dibujar conexiones
    for (let i = 0; i < positions.length - 1; i++) {
        const currentLayer = positions[i];
        const nextLayer = positions[i + 1];
        for (let a = 0; a < currentLayer.length; a++) {
            for (let b = 0; b < nextLayer.length; b++) {
                ctx.beginPath();
                ctx.moveTo(currentLayer[a].x, currentLayer[a].y);
                ctx.lineTo(nextLayer[b].x, nextLayer[b].y);
                ctx.strokeStyle = '#ccc';
                ctx.stroke();
            }
        }
    }

    // Generar descripción de la red
    const description = `La red tiene ${layers.length} capas: `;
    let layerDescription = layers.map((n, idx) => `Capa ${idx + 1}: ${n} neuronas`).join(', ');
    document.getElementById('networkDescription').innerText = description + layerDescription;

    // Calcular número de parámetros
    let parameters = 0;
    for (let i = 0; i < layers.length - 1; i++) {
        parameters += layers[i] * layers[i + 1];
    }
    document.getElementById('parameterCount').innerText = `Número total de parámetros (pesos): ${parameters}`;
}

// Dibujar la red inicial
drawNetwork();
