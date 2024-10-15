// script.js

document.addEventListener('DOMContentLoaded', () => {
    const imageSize = 10;
    let imageMatrix = [];
    let filterMatrix = [];
    let filterSize = 3;
    let stride = 1;

    const imageContainer = document.getElementById('image-container');
    const filterContainer = document.getElementById('filter-container');
    const filterSizeSelect = document.getElementById('filter-size');
    const strideInput = document.getElementById('stride');
    const applyButton = document.getElementById('apply-button');
    const resetButton = document.getElementById('reset-button');

    function generateImageMatrix() {
        imageMatrix = [];
        for (let i = 0; i < imageSize; i++) {
            let row = [];
            for (let j = 0; j < imageSize; j++) {
                row.push(Math.random().toFixed(2));
            }
            imageMatrix.push(row);
        }
        renderImageMatrix();
    }

    function renderImageMatrix() {
        imageContainer.innerHTML = '';
        const grid = document.createElement('div');
        grid.className = 'grid';
        imageMatrix.forEach(row => {
            row.forEach(value => {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.textContent = value;
                grid.appendChild(cell);
            });
        });
        imageContainer.appendChild(grid);
    }

    function generateFilterMatrix() {
        filterMatrix = [];
        for (let i = 0; i < filterSize; i++) {
            let row = [];
            for (let j = 0; j < filterSize; j++) {
                row.push(0);
            }
            filterMatrix.push(row);
        }
        renderFilterMatrix();
    }

    function renderFilterMatrix() {
        filterContainer.innerHTML = '';
        const grid = document.createElement('div');
        grid.className = 'grid';
        grid.style.setProperty('--filter-size', filterSize);
        filterMatrix.forEach((row, i) => {
            row.forEach((value, j) => {
                const cell = document.createElement('input');
                cell.type = 'number';
                cell.value = value;
                cell.className = 'grid-cell';
                cell.style.padding = '0';
                cell.addEventListener('change', (e) => {
                    filterMatrix[i][j] = parseFloat(e.target.value);
                });
                grid.appendChild(cell);
            });
        });
        filterContainer.appendChild(grid);
    }

    function applyConvolution() {
        const outputSize = Math.floor((imageSize - filterSize) / stride) + 1;
        let outputMatrix = [];
        for (let i = 0; i < outputSize; i++) {
            let row = [];
            for (let j = 0; j < outputSize; j++) {
                let sum = 0;
                for (let fi = 0; fi < filterSize; fi++) {
                    for (let fj = 0; fj < filterSize; fj++) {
                        const imageVal = parseFloat(imageMatrix[i * stride + fi][j * stride + fj]);
                        const filterVal = filterMatrix[fi][fj];
                        sum += imageVal * filterVal;
                    }
                }
                row.push(sum.toFixed(2));
            }
            outputMatrix.push(row);
        }
        imageMatrix = outputMatrix;
        renderImageMatrix();
    }

    filterSizeSelect.addEventListener('change', () => {
        filterSize = parseInt(filterSizeSelect.value);
        generateFilterMatrix();
    });

    strideInput.addEventListener('change', () => {
        stride = parseInt(strideInput.value);
    });

    applyButton.addEventListener('click', () => {
        applyConvolution();
    });

    resetButton.addEventListener('click', () => {
        generateImageMatrix();
        generateFilterMatrix();
    });

    // Inicialización
    generateImageMatrix();
    generateFilterMatrix();
});
