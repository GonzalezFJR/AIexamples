document.addEventListener('DOMContentLoaded', function() {
    // Obtener elementos del DOM
    const imageSelect = document.getElementById('imageSelect');
    const imageUpload = document.getElementById('imageUpload');
    const modelSelect = document.getElementById('modelSelect');
    const layerSelect = document.getElementById('layerSelect');
    const generateButton = document.getElementById('generateButton');
    const originalImage = document.getElementById('originalImage');
    const activationImage = document.getElementById('activationImage');
    const transparencySlider = document.getElementById('transparencySlider');

    // Obtener custom_id de la URL si existe
    const urlParams = new URLSearchParams(window.location.search);
    const customId = urlParams.get('custom_id');

    // Cargar las imágenes de ejemplo
    fetch('/activation_maps/images')
        .then(response => response.json())
        .then(data => {
            data.images.forEach(imageName => {
                const option = document.createElement('option');
                option.value = imageName;
                option.text = imageName;
                imageSelect.add(option);
            });
        });

    // Cargar los modelos disponibles
    fetch('/activation_maps/models')
        .then(response => response.json())
        .then(data => {
            data.models.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                option.text = modelName;
                modelSelect.add(option);
            });
            loadLayers();
        });

    // Actualizar las capas cuando se seleccione un modelo
    modelSelect.addEventListener('change', loadLayers);

    function loadLayers() {
        const modelName = modelSelect.value;
        layerSelect.innerHTML = ''; // Limpiar las opciones
        if (modelName === 'customs') {
            fetch(`/activation_maps/layers/custom`)
                .then(response => response.json())
                .then(data => {
                    data.layers.forEach(layer => {
                        const option = document.createElement('option');
                        option.value = layer.idx;
                        option.text = `Capa ${layer.idx}: ${layer.name} (${layer.type})`;
                        layerSelect.add(option);
                    });
                });
        } else {
            fetch(`/activation_maps/layers/${modelName}`)
                .then(response => response.json())
                .then(data => {
                    data.layers.forEach(layer => {
                        const option = document.createElement('option');
                        option.value = layer.idx;
                        option.text = `Capa ${layer.idx}: ${layer.name} (${layer.type})`;
                        layerSelect.add(option);
                    });
                });
        }
    }

    generateButton.addEventListener('click', function() {
        const modelName = modelSelect.value;
        const layerIdx = layerSelect.value;
        const formData = new FormData();

        if (modelName === 'custom') {
            formData.append('model_name', modelName);
            formData.append('layer_idx', layerIdx);
            if (customId) {
                formData.append('custom_id', customId);
            }
        } else {
            if (imageUpload.files && imageUpload.files[0]) {
                formData.append('image_file', imageUpload.files[0]);
            } else {
                const imageName = imageSelect.value;
                formData.append('image_name', imageName);
            }
            formData.append('model_name', modelName);
            formData.append('layer_idx', layerIdx);
        }

        fetch('/activation_maps/activation_map', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return response.json().then(err => { throw err; });
            }
        })
        .then(data => {
            const originalImageSrc = 'data:image/png;base64,' + data.original_image;
            const activationImageSrc = 'data:image/png;base64,' + data.activation_map;
            originalImage.src = originalImageSrc;
            activationImage.src = activationImageSrc;
            activationImage.style.opacity = transparencySlider.value;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Ocurrió un error al generar el mapa de activación');
        });
    });

    // Actualizar la transparencia del mapa de activación
    transparencySlider.addEventListener('input', function() {
        activationImage.style.opacity = transparencySlider.value;
    });
});