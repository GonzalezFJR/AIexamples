// Datos de las funciones de activación
const activationFunctions = [
    {
        name: "1. Sigmoid (Logística)",
        formula: "f(x) = \\frac{1}{1 + e^{-x}}",
        description: "Transforma valores a un rango entre 0 y 1. Es útil para problemas binarios, pero tiene el problema de saturación y desvanecimiento del gradiente.",
        usage: "Salidas de clasificación binaria.",
        function: (x) => 1 / (1 + Math.exp(-x)),
        domain: [-10, 10]
    },
    {
        name: "2. Tanh (Tangente Hiperbólica)",
        formula: "f(x) = \\tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}",
        description: "Similar a la Sigmoid, pero su rango es de -1 a 1, lo que generalmente mejora el entrenamiento.",
        usage: "Redes recurrentes (RNNs).",
        function: (x) => Math.tanh(x),
        domain: [-5, 5]
    },
    {
        name: "3. ReLU (Rectified Linear Unit)",
        formula: "f(x) = \\max(0, x)",
        description: "Devuelve el valor si es positivo, de lo contrario devuelve 0. Es simple y eficiente, pero puede causar el problema de neuronas 'muertas' cuando los gradientes se vuelven cero.",
        usage: "Redes profundas y convolucionales.",
        function: (x) => Math.max(0, x),
        domain: [-10, 10]
    },
    {
        name: "4. Leaky ReLU",
        formula: `f(x) = \\begin{cases} x & \\text{si } x > 0 \\\\ \\alpha x & \\text{si } x \\leq 0 \\end{cases} \\quad (\\alpha = 0.01)`,
        description: "Variante de ReLU que permite un pequeño gradiente cuando x ≤ 0, solucionando el problema de neuronas muertas.",
        usage: "Mejoras en redes convolucionales.",
        function: (x) => x > 0 ? x : 0.01 * x,
        domain: [-10, 10]
    },
    {
        name: "5. Parametric ReLU (PReLU)",
        formula: `f(x) = \\begin{cases} x & \\text{si } x > 0 \\\\ \\alpha x & \\text{si } x \\leq 0 \\end{cases} \\quad (\\alpha \\text{ aprendible})`,
        description: "Similar a Leaky ReLU, pero con un valor de α aprendible durante el entrenamiento.",
        usage: "Redes profundas donde se quiere flexibilidad en el aprendizaje de la función de activación.",
        function: (x) => x > 0 ? x : 0.1 * x, // α es aprendible, aquí usamos α = 0.1 para ilustrar
        domain: [-10, 10]
    },
    {
        name: "6. Exponential Linear Unit (ELU)",
        formula: `f(x) = \\begin{cases} x & \\text{si } x > 0 \\\\ \\alpha (e^{x} - 1) & \\text{si } x \\leq 0 \\end{cases} \\quad (\\alpha = 1)`,
        description: "ELU tiene la ventaja de tener una salida negativa, lo que ayuda a mantener activaciones cercanas a 0 y mejorar la convergencia.",
        usage: "Redes profundas y convolucionales.",
        function: (x) => x >= 0 ? x : (Math.exp(x) - 1), // α = 1
        domain: [-5, 5]
    },
    {
        name: "7. Scaled Exponential Linear Unit (SELU)",
        formula: `f(x) = \\lambda \\begin{cases} x & \\text{si } x > 0 \\\\ \\alpha (e^{x} - 1) & \\text{si } x \\leq 0 \\end{cases} \\quad (\\alpha \\approx 1.673, \\lambda \\approx 1.0507)`,
        description: "Una variante de ELU escalada automáticamente para redes auto-normalizables.",
        usage: "Redes profundas con auto-normalización.",
        function: (x) => {
            const alpha = 1.67326;
            const lambda = 1.0507;
            return x >= 0 ? lambda * x : lambda * alpha * (Math.exp(x) - 1);
        },
        domain: [-5, 5]
    },
    {
        name: "8. Softmax",
        formula: "f(x_i) = \\frac{e^{x_i}}{\\sum_{j} e^{x_j}}",
        description: "Convierte un vector de valores en probabilidades, donde la suma de todas las probabilidades es 1.",
        usage: "Clasificación multiclase.",
        function: (x) => {
            // Softmax se aplica a vectores; aquí mostramos una representación simplificada
            return 1; // Resultado constante para ilustración
        },
        domain: [-10, 10]
    },
    {
        name: "9. Swish",
        formula: "f(x) = x \\cdot \\text{sigmoid}(x) = x \\cdot \\frac{1}{1 + e^{-x}}",
        description: "Función introducida por Google, similar a la ReLU pero suavizada, con mejores propiedades para el flujo de gradiente.",
        usage: "Redes profundas, en particular en modelos como EfficientNet.",
        function: (x) => x / (1 + Math.exp(-x)),
        domain: [-10, 10]
    },
    {
        name: "10. Mish",
        formula: "f(x) = x \\cdot \\tanh(\\ln(1 + e^{x}))",
        description: "Propuesta más reciente, es similar a Swish pero ofrece una curva más suave y mejores características en tareas de clasificación y segmentación.",
        usage: "Redes profundas, redes convolucionales.",
        function: (x) => x * Math.tanh(Math.log(1 + Math.exp(x))),
        domain: [-10, 10]
    },
    {
        name: "11. Softplus",
        formula: "f(x) = \\ln(1 + e^{x})",
        description: "Aproximación suave de la ReLU, con el beneficio de ser siempre diferenciable.",
        usage: "Redes profundas.",
        function: (x) => Math.log(1 + Math.exp(x)),
        domain: [-10, 10]
    },
    {
        name: "12. Softsign",
        formula: "f(x) = \\frac{x}{1 + |x|}",
        description: "Similar a la función tanh, pero más suave y sin exponenciales.",
        usage: "Redes profundas, aunque menos común.",
        function: (x) => x / (1 + Math.abs(x)),
        domain: [-10, 10]
    },
    {
        name: "13. Maxout",
        formula: "f(x) = \\max(w_1 x + b_1, w_2 x + b_2)",
        description: "Generaliza la ReLU y la Leaky ReLU, seleccionando el máximo de dos funciones lineales.",
        usage: "Redes profundas.",
        function: (x) => Math.max(x, 0.5 * x), // Simplificado para ilustración
        domain: [-10, 10]
    },
    {
        name: "14. GELU (Gaussian Error Linear Unit)",
        formula: "f(x) = 0.5 x \\left[1 + \\tanh \\left( \\sqrt{\\frac{2}{\\pi}} (x + 0.044715 x^3) \\right) \\right]",
        description: "Introduce propiedades gaussianas, similar a Swish pero con una justificación probabilística.",
        usage: "Modelos recientes como BERT y Transformers.",
        function: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
        domain: [-10, 10]
    },
    {
        name: "15. Hard Sigmoid",
        formula: "f(x) = \\max(0, \\min(1, 0.2 x + 0.5))",
        description: "Aproximación simplificada de Sigmoid, útil en dispositivos con poca capacidad computacional.",
        usage: "Redes ligeras o móviles.",
        function: (x) => Math.max(0, Math.min(1, 0.2 * x + 0.5)),
        domain: [-10, 10]
    },
    {
        name: "16. Hard Swish",
        formula: "f(x) = x \\cdot \\frac{\\max(0, \\min(6, x + 3))}{6}",
        description: "Aproximación de Swish que es más eficiente para implementaciones en hardware.",
        usage: "Redes móviles como MobileNet.",
        function: (x) => x * Math.max(0, Math.min(6, x + 3)) / 6,
        domain: [-10, 10]
    },
    {
        name: "17. Binary Step",
        formula: `f(x) = \\begin{cases} 1 & \\text{si } x \\geq 0 \\\\ 0 & \\text{si } x < 0 \\end{cases}`,
        description: "Devuelve 1 o 0 dependiendo del signo del input, pero no es diferenciable, por lo que no se usa en el aprendizaje profundo.",
        usage: "Redes neuronales binarias y modelos teóricos.",
        function: (x) => x >= 0 ? 1 : 0,
        domain: [-10, 10]
    },
    {
        name: "18. Rectified Power Unit (RePU)",
        formula: "f(x) = \\max(0, x^n), \\quad n = 2",
        description: "Una generalización de ReLU con una potencia n, lo que puede darle más flexibilidad.",
        usage: "Variantes en redes convolucionales.",
        function: (x) => Math.max(0, Math.pow(x, 2)), // n = 2
        domain: [-5, 5]
    },
    {
        name: "19. Sinusoidal Activation Function",
        formula: "f(x) = \\sin(x)",
        description: "Usa la función seno como activación, lo que puede ayudar en tareas donde las señales periódicas son importantes.",
        usage: "Modelos especializados en datos periódicos.",
        function: (x) => Math.sin(x),
        domain: [-2 * Math.PI, 2 * Math.PI]
    },
    {
        name: "20. ArcTan",
        formula: "f(x) = \\arctan(x)",
        description: "Similar a la función Sigmoid pero con un rango entre $-\\frac{\\pi}{2}$ y $\\frac{\\pi}{2}$. ",
        usage: "Redes neuronales en contextos matemáticos especializados.",
        function: (x) => Math.atan(x),
        domain: [-10, 10]
    }
];

// Inicializar el selector de funciones
const selectElement = document.getElementById('activation-function');
activationFunctions.forEach((func, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.text = func.name;
    selectElement.add(option);
});

// Configuración inicial
let chart;
selectElement.selectedIndex = 0;
updateFunction(selectElement.value);

// Actualizar cuando se selecciona una nueva función
selectElement.addEventListener('change', function() {
    updateFunction(this.value);
});

function updateFunction(index) {
    const func = activationFunctions[index];
    // Actualizar la fórmula y la descripción
    document.getElementById('formula').innerHTML = '$$' + func.formula + '$$';
    document.getElementById('description').textContent = func.description;
    document.getElementById('usage').textContent = func.usage;

    // Pedir a MathJax que procese las nuevas fórmulas
    MathJax.typesetPromise();

    // Generar datos para la gráfica
    const xValues = [];
    const yValues = [];
    const xmin = func.domain[0];
    const xmax = func.domain[1];
    const step = (xmax - xmin) / 200; // Más puntos para una gráfica suave
    for (let x = xmin; x <= xmax; x += step) {
        xValues.push(x);
        yValues.push(func.function(x));
    }

    // Crear o actualizar la gráfica
    if (chart) {
        chart.data.labels = xValues;
        chart.data.datasets[0].data = yValues;
        chart.data.datasets[0].label = func.name;
        chart.update();
    } else {
        const ctx = document.getElementById('activationChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: xValues,
                datasets: [{
                    label: func.name,
                    data: yValues,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false,
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'x'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'f(x)'
                        }
                    }
                }
            }
        });
    }
}
