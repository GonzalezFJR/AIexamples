let currentMatrix = [];
let displayMatrix = [];
let operationsApplied = false;
let previewEnabled = false;

document.addEventListener("DOMContentLoaded", function() {
    // Cargar y renderizar la matriz inicial
    fetchMatrix();

    // Actualizar la matriz cada 5 segundos si no se han aplicado operaciones
    setInterval(function() {
        if (!operationsApplied) {
            fetchMatrix();
        }
    }, 5000);

    // Event listener para mostrar/ocultar valores de los píxeles
    let showValues = true;
    document.getElementById('toggle-values-button').addEventListener('click', function() {
        showValues = !showValues;
        togglePixelValues(showValues);
        this.textContent = showValues ? 'Ocultar valores' : 'Mostrar valores';
    });

    // Event listener para normalizar la matriz
    document.getElementById('normalize-button').addEventListener('click', function() {
        normalizeMatrix();
    });

    // Event listener para restablecer la matriz
    document.getElementById('reset-button').addEventListener('click', function() {
        operationsApplied = false;
        clearMessage();
        fetchMatrix();
    });

    // Event listener para generar el filtro
    document.getElementById('generate-filter-button').addEventListener('click', function() {
        generateFilterMatrix();
    });

    // Event listener para aplicar el filtro convolucional
    document.getElementById('apply-filter-button').addEventListener('click', function() {
        applyConvolution();
    });

    // Event listener para aplicar pooling
    document.getElementById('apply-pooling-button').addEventListener('click', function() {
        applyPooling();
    });

    // Event listener para filtros predefinidos
    document.getElementById('predefined-filters').addEventListener('change', function() {
        let selectedFilter = this.value;
        let M = parseInt(document.getElementById('filter-size').value);
        if (selectedFilter !== "") {
            generatePredefinedFilter(selectedFilter, M);
        }
    });

    // Event listener para activar/desactivar previsualización
    document.getElementById('preview-toggle').addEventListener('change', function() {
        previewEnabled = this.checked;
        if (previewEnabled) {
            enablePreview();
        } else {
            disablePreview();
        }
    });
});

// Función para mostrar mensajes
function showMessage(text, type="is-danger") {
    let messageDiv = document.getElementById('message');
    messageDiv.innerHTML = `<div class="notification ${type}">
                                <button class="delete"></button>
                                ${text}
                            </div>`;

    // Añadir event listener al botón de cerrar
    messageDiv.querySelector('.delete').addEventListener('click', function() {
        messageDiv.innerHTML = '';
    });
}

function clearMessage() {
    let messageDiv = document.getElementById('message');
    messageDiv.innerHTML = '';
}

// Función para obtener la matriz del servidor
function fetchMatrix() {
    fetch('/get_matrix')
    .then(response => response.json())
    .then(data => {
        currentMatrix = data.matrix;
        if (previewEnabled) {
            let padding = parseInt(document.getElementById('padding').value);
            displayMatrix = padMatrix(currentMatrix, padding);
        } else {
            displayMatrix = currentMatrix;
        }
        renderMatrix(displayMatrix);
    })
    .catch(error => console.error('Error:', error));
}

// Función para renderizar la matriz en el DOM
function renderMatrix(matrix) {
    let N = matrix.length;
    let container = document.getElementById("matrix-container");
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${N}, 1fr)`;

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let value = matrix[i][j];
            let pixel = document.createElement("div");
            pixel.className = "pixel";
            pixel.style.backgroundColor = `rgb(${clamp(value)}, ${clamp(value)}, ${clamp(value)})`;

            let p = document.createElement("p");
            p.textContent = Math.floor(value); // Mostrar solo la parte entera

            // Almacenar la posición en data attributes
            pixel.dataset.row = i;
            pixel.dataset.col = j;

            pixel.appendChild(p);
            container.appendChild(pixel);
        }
    }

    // Si la previsualización está habilitada, añadir eventos a los píxeles
    if (previewEnabled) {
        addPreviewEventListeners();
    }
}

// Función para mostrar/ocultar valores de los píxeles
function togglePixelValues(show) {
    let container = document.getElementById("matrix-container");
    if (show) {
        container.classList.remove('hidden-values');
    } else {
        container.classList.add('hidden-values');
    }
}

// Función para normalizar la matriz
function normalizeMatrix() {
    let needsNormalization = false;
    let minVal = Infinity;
    let maxVal = -Infinity;

    // Encontrar el mínimo y máximo valor
    for (let row of currentMatrix) {
        for (let val of row) {
            if (val < 0 || val > 255) {
                needsNormalization = true;
            }
            minVal = Math.min(minVal, val);
            maxVal = Math.max(maxVal, val);
        }
    }

    if (!needsNormalization) {
        // La matriz ya está normalizada, no hacemos nada
        showMessage('La matriz ya está normalizada.', 'is-info');
        return;
    }

    // Normalizar la matriz
    for (let i = 0; i < currentMatrix.length; i++) {
        for (let j = 0; j < currentMatrix[i].length; j++) {
            currentMatrix[i][j] = ((currentMatrix[i][j] - minVal) / (maxVal - minVal)) * 255;
        }
    }

    // Actualizar displayMatrix si la previsualización está habilitada
    if (previewEnabled) {
        let padding = parseInt(document.getElementById('padding').value);
        displayMatrix = padMatrix(currentMatrix, padding);
    } else {
        displayMatrix = currentMatrix;
    }

    renderMatrix(displayMatrix);
    showMessage('Matriz normalizada.', 'is-success');
}

// Función para generar la matriz del filtro
function generateFilterMatrix() {
    let M = parseInt(document.getElementById('filter-size').value);
    let filterContainer = document.getElementById('filter-container');
    filterContainer.innerHTML = '';
    filterContainer.style.gridTemplateColumns = `repeat(${M}, 1fr)`;

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
            let cell = document.createElement('div');
            cell.className = 'filter-cell';

            let input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01'; // Permitir hasta dos decimales
            input.value = '0.00'; // Mostrar hasta dos decimales

            cell.appendChild(input);
            filterContainer.appendChild(cell);
        }
    }
}

// Función para generar filtros predefinidos
function generatePredefinedFilter(filterName, M) {
    let filterValues = [];
    // Definir los filtros predefinidos
    if (filterName === 'edge-detection') {
        // Por ejemplo, un filtro Laplaciano
        if (M === 3) {
            filterValues = [0, -1, 0,
                            -1, 4, -1,
                            0, -1, 0];
        } else {
            showMessage('El filtro de detección de bordes está definido para tamaño 3x3. Cambie el tamaño del filtro a 3.', 'is-warning');
            return;
        }
    } else if (filterName === 'sharpen') {
        // Filtro de enfoque
        if (M === 3) {
            filterValues = [0, -1, 0,
                            -1, 5, -1,
                            0, -1, 0];
        } else {
            showMessage('El filtro de enfoque está definido para tamaño 3x3. Cambie el tamaño del filtro a 3.', 'is-warning');
            return;
        }
    } else if (filterName === 'blur') {
        // Filtro de desenfoque
        if (M === 3) {
            filterValues = [1/9, 1/9, 1/9,
                            1/9, 1/9, 1/9,
                            1/9, 1/9, 1/9];
        } else {
            showMessage('El filtro de desenfoque está definido para tamaño 3x3. Cambie el tamaño del filtro a 3.', 'is-warning');
            return;
        }
    }

    // Generar la matriz del filtro con los valores predefinidos
    let filterContainer = document.getElementById('filter-container');
    filterContainer.innerHTML = '';
    filterContainer.style.gridTemplateColumns = `repeat(${M}, 1fr)`;

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
            let cell = document.createElement('div');
            cell.className = 'filter-cell';

            let input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01'; // Permitir hasta dos decimales
            input.value = parseFloat(filterValues[i * M + j]).toFixed(2); // Mostrar hasta dos decimales

            cell.appendChild(input);
            filterContainer.appendChild(cell);
        }
    }
}

// Función para aplicar la convolución
function applyConvolution() {
    let filterCells = document.querySelectorAll('.filter-cell input');
    if (filterCells.length === 0) {
        showMessage('Por favor, genera el filtro primero.', 'is-warning');
        return;
    }

    let M = parseInt(document.getElementById('filter-size').value);
    let stride = parseInt(document.getElementById('stride').value);
    let padding = parseInt(document.getElementById('padding').value);

    // Obtener valores del filtro
    let filter = [];
    for (let input of filterCells) {
        filter.push(parseFloat(parseFloat(input.value).toFixed(2))); // Asegurar dos decimales
    }

    // Verificar si la matriz está disponible
    if (currentMatrix.length === 0) {
        showMessage('La matriz no está disponible.', 'is-danger');
        return;
    }

    let result = convolveMatrix(currentMatrix, filter, M, stride, padding);

    if (result) {
        currentMatrix = result; // Actualizar la matriz actual
        displayMatrix = currentMatrix;
        operationsApplied = true; // Marcar que se ha aplicado una operación
        previewEnabled = false; // Desactivar previsualización después de aplicar
        document.getElementById('preview-toggle').checked = false;
        clearMessage();
        renderMatrix(displayMatrix);
    } else {
        showMessage('Las dimensiones no son adecuadas para aplicar la convolución con los parámetros dados.', 'is-warning');
    }
}

// Función para realizar la convolución
function convolveMatrix(matrix, filter, M, stride, padding) {
    let N = matrix.length;
    let paddedMatrix = padMatrix(matrix, padding);
    let newSize = paddedMatrix.length;
    let outputSize = Math.floor((newSize - M) / stride) + 1;
    if (outputSize <= 0) {
        return null;
    }

    let result = [];
    for (let i = 0; i < outputSize; i++) {
        result.push([]);
        for (let j = 0; j < outputSize; j++) {
            let sum = 0;
            for (let m = 0; m < M; m++) {
                for (let n = 0; n < M; n++) {
                    let x = i * stride + m;
                    let y = j * stride + n;
                    sum += paddedMatrix[x][y] * filter[m * M + n];
                }
            }
            result[i].push(sum); // No redondear aún para permitir valores fuera de 0-255
        }
    }
    return result;
}

// Función para agregar padding a la matriz
function padMatrix(matrix, padding) {
    let N = matrix.length;
    let newSize = N + 2 * padding;
    let paddedMatrix = [];

    for (let i = 0; i < newSize; i++) {
        paddedMatrix.push([]);
        for (let j = 0; j < newSize; j++) {
            if (i < padding || i >= N + padding || j < padding || j >= N + padding) {
                paddedMatrix[i].push(0);
            } else {
                paddedMatrix[i].push(matrix[i - padding][j - padding]);
            }
        }
    }
    return paddedMatrix;
}

// Función para aplicar pooling
function applyPooling() {
    let poolingType = document.getElementById('pooling-type').value;
    let poolingSize = parseInt(document.getElementById('pooling-size').value);
    let poolingStride = parseInt(document.getElementById('pooling-stride').value);

    // Verificar si la matriz está disponible
    if (currentMatrix.length === 0) {
        showMessage('La matriz no está disponible.', 'is-danger');
        return;
    }

    let result = poolMatrix(currentMatrix, poolingSize, poolingStride, poolingType);

    if (result) {
        currentMatrix = result; // Actualizar la matriz actual
        displayMatrix = currentMatrix;
        operationsApplied = true; // Marcar que se ha aplicado una operación
        previewEnabled = false; // Desactivar previsualización después de aplicar
        document.getElementById('preview-toggle').checked = false;
        clearMessage();
        renderMatrix(displayMatrix);
    } else {
        showMessage('Las dimensiones no son adecuadas para aplicar el pooling con los parámetros dados.', 'is-warning');
    }
}

// Función para realizar pooling
function poolMatrix(matrix, K, stride, poolingType) {
    let N = matrix.length;
    let outputSize = Math.floor((N - K) / stride) + 1;
    if (outputSize <= 0) {
        return null;
    }

    let result = [];
    for (let i = 0; i < outputSize; i++) {
        result.push([]);
        for (let j = 0; j < outputSize; j++) {
            let pool = [];
            for (let m = 0; m < K; m++) {
                for (let n = 0; n < K; n++) {
                    let x = i * stride + m;
                    let y = j * stride + n;
                    pool.push(matrix[x][y]);
                }
            }
            let value;
            if (poolingType === 'max') {
                value = Math.max(...pool);
            } else if (poolingType === 'average') {
                value = pool.reduce((a, b) => a + b, 0) / pool.length;
            }
            result[i].push(value);
        }
    }
    return result;
}

// Función auxiliar para limitar los valores entre 0 y 255
function clamp(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
}

// Función para habilitar la previsualización
function enablePreview() {
    let padding = parseInt(document.getElementById('padding').value);
    displayMatrix = padMatrix(currentMatrix, padding);
    renderMatrix(displayMatrix);
}

// Función para deshabilitar la previsualización
function disablePreview() {
    displayMatrix = currentMatrix;
    renderMatrix(displayMatrix);
}

// Función para añadir eventos de previsualización a los píxeles
function addPreviewEventListeners() {
    let pixels = document.querySelectorAll('#matrix-container .pixel');
    pixels.forEach((pixel) => {
        pixel.addEventListener('mouseenter', previewConvolution);
        pixel.addEventListener('mouseleave', clearPreview);
    });
}

// Función para previsualizar la convolución
function previewConvolution(event) {
    let pixel = event.currentTarget;
    let i = parseInt(pixel.dataset.row);
    let j = parseInt(pixel.dataset.col);

    let M = parseInt(document.getElementById('filter-size').value);
    let padding = parseInt(document.getElementById('padding').value);

    // Obtener valores del filtro
    let filterCells = document.querySelectorAll('.filter-cell input');
    if (filterCells.length === 0) {
        // No hay filtro generado, no podemos previsualizar
        return;
    }
    let filter = [];
    for (let input of filterCells) {
        filter.push(parseFloat(parseFloat(input.value).toFixed(2))); // Asegurar dos decimales
    }

    let halfM = Math.floor(M / 2);

    // Preparar para resaltar la región MxM
    let pixelsToHighlight = [];

    let sum = 0;

    for (let m = -halfM; m <= halfM; m++) {
        for (let n = -halfM; n <= halfM; n++) {
            let x = i + m;
            let y = j + n;

            if (x >= 0 && x < displayMatrix.length && y >= 0 && y < displayMatrix.length) {
                let value = displayMatrix[x][y];
                let filterValue = filter[(m + halfM) * M + (n + halfM)];
                sum += value * filterValue;

                // Encontrar el elemento DOM del píxel para resaltar
                let selector = `.pixel[data-row="${x}"][data-col="${y}"]`;
                let pixelToHighlight = document.querySelector(selector);
                if (pixelToHighlight) {
                    pixelsToHighlight.push(pixelToHighlight);
                }
            }
        }
    }

    // Resaltar los píxeles
    pixelsToHighlight.forEach(p => {
        p.classList.add('highlight');
    });

    // Mostrar el valor calculado en el píxel central
    let computedValue = sum.toFixed(2);
    let overlay = document.createElement('div');
    overlay.className = 'overlay';
    overlay.textContent = computedValue;
    pixel.appendChild(overlay);
}

// Función para limpiar la previsualización
function clearPreview(event) {
    // Eliminar los resaltados
    let highlightedPixels = document.querySelectorAll('.pixel.highlight');
    highlightedPixels.forEach(p => {
        p.classList.remove('highlight');
    });

    // Eliminar las superposiciones
    let overlays = document.querySelectorAll('.pixel .overlay');
    overlays.forEach(overlay => {
        overlay.parentNode.removeChild(overlay);
    });
}
