// static/board.js

const apiUrl = "http://127.0.0.1:8000"; // Ajusta según tu backend
let actionCount = 0;         // Contador de acciones global
let agentIntervalId = null;  // Para parar el bucle del agente si termina

// ==== Semillas globales (para reutilizar la que se generó al entrenar) ====
let dqnSeed = null;
let classicSeed = null;

// ========== ENTRENAR MODELO DQN ==========
async function trainDQN() {
  const episodes = parseInt(document.getElementById("dqnEpisodesInput").value) || 1000;
  showMessage(`Entrenando DQN con ${episodes} episodios...`);
  try {
    const resp = await fetch(`${apiUrl}/train_dqn?episodes=${episodes}`);
    const data = await resp.json();
    if (data.error) {
      showMessage("Error en entrenamiento DQN: " + data.error);
      return;
    }
    showMessage(data.message || "Entrenamiento completado.");
    // Guardar la semilla que devolvió
    if (data.seed !== undefined) {
      dqnSeed = data.seed;
      console.log("DQN entrenado con seed=", dqnSeed);
    }
  } catch (err) {
    console.error(err);
    showMessage("Error llamando a /train_dqn.");
  }
}

// ========== ENTRENAR MODELO CLÁSICO ==========
async function trainClassic() {
  const episodes = parseInt(document.getElementById("classicEpisodesInput").value) || 2000;
  showMessage(`Entrenando Q-Learning clásico con ${episodes} episodios...`);
  try {
    const resp = await fetch(`${apiUrl}/train_classic?episodes=${episodes}`);
    const data = await resp.json();
    if (data.error) {
      showMessage("Error en entrenamiento clásico: " + data.error);
      return;
    }
    showMessage(data.message || "Entrenamiento clásico completado.");
    // Guardar la semilla
    if (data.seed !== undefined) {
      classicSeed = data.seed;
      console.log("Clásico entrenado con seed=", classicSeed);
    }
  } catch (err) {
    console.error(err);
    showMessage("Error llamando a /train_classic.");
  }
}


// ========== LÓGICA DE PARTIDA HUMANA ==========

async function newGame(n = 5, d = 4) {
  try {
    const response = await fetch(`${apiUrl}/new_game?n=${n}&d=${d}`);
    const gameState = await response.json();
    // Resetear contador de acciones
    actionCount = 0;
    updateActionCount();
    renderBoard(gameState);
    showMessage("Partida nueva. Usa las flechas para moverte.");
  } catch (error) {
    console.error("Error al crear nuevo juego:", error);
    showMessage("Error al crear nuevo juego.");
  }
}

async function movePlayer(direction) {
  try {
    // Incrementamos el contador de acciones del usuario
    actionCount++;
    updateActionCount();

    const response = await fetch(`${apiUrl}/move?direction=${direction}`, {
      method: "POST"
    });
    const gameState = await response.json();
    renderBoard(gameState);

    if (gameState.status === "lost") {
      showMessage("Has perdido. ¡Intenta de nuevo!");
    } else if (gameState.status === "won") {
      showMessage("¡Victoria!");
    }
  } catch (error) {
    console.error("Error al mover el jugador:", error);
  }
}

// Detectar flechas para la partida humana
document.addEventListener("keydown", (e) => {
  // Flechas: ArrowUp, ArrowDown, ArrowLeft, ArrowRight
  switch (e.key) {
    case "ArrowUp":
      movePlayer("up");
      break;
    case "ArrowDown":
      movePlayer("down");
      break;
    case "ArrowLeft":
      movePlayer("left");
      break;
    case "ArrowRight":
      movePlayer("right");
      break;
    default:
      break;
  }
});


// ========== PARTIDA DEL AGENTE DQN ==========

async function startAgentPlay() {
  // Cancelar si ya hay un intervalo corriendo
  if (agentIntervalId) {
    clearInterval(agentIntervalId);
    agentIntervalId = null;
  }

  // Llamamos a /start_play, pasando la semilla si tenemos dqnSeed
  let url = `${apiUrl}/start_play`;
  if (dqnSeed !== null) {
    url += `?seed=${dqnSeed}`;
  }

  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (data.error) {
      showMessage(data.error);
      return;
    }
    // Resetear contador de acciones
    actionCount = 0;
    updateActionCount();

    renderBoard(data);
    showMessage("Partida del agente DQN iniciada. Observa los movimientos...");

    // Ahora cada 0.5s llamamos a /play_step
    agentIntervalId = setInterval(async () => {
      const stepRes = await fetch(`${apiUrl}/play_step`);
      const stepData = await stepRes.json();

      if (stepData.error) {
        showMessage(stepData.error);
        clearInterval(agentIntervalId);
        agentIntervalId = null;
        return;
      }

      // Actualizar tablero
      renderBoard(stepData.board_info);

      // Incrementar contador de acciones
      actionCount++;
      updateActionCount();

      // Comprobar si ha terminado
      if (stepData.done) {
        clearInterval(agentIntervalId);
        agentIntervalId = null;
        if (stepData.board_info.status === "won") {
          showMessage("¡El agente DQN ha ganado!");
        } else {
          showMessage("El agente DQN ha perdido...");
        }
      }
    }, 500);

  } catch (err) {
    console.error("Error al iniciar partida del agente DQN:", err);
  }
}


// ========== PARTIDA Q-LEARNING CLÁSICO ==========

async function startClassicPlay() {
  // Cancelar si ya hay un intervalo corriendo
  if (agentIntervalId) {
    clearInterval(agentIntervalId);
    agentIntervalId = null;
  }

  // Similar a DQN, llamamos a /start_classic, con la semilla
  let url = `${apiUrl}/start_classic`;
  if (classicSeed !== null) {
    url += `?seed=${classicSeed}`;
  }

  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (data.error) {
      showMessage(data.error);
      return;
    }
    actionCount = 0;
    renderBoard(data);
    showMessage("Partida Q-Learning clásico iniciada...");

    // Hacemos pasos cada 0.5s
    agentIntervalId = setInterval(async () => {
      const stepResp = await fetch(`${apiUrl}/play_step_classic`);
      const stepData = await stepResp.json();
      if (stepData.error) {
        showMessage(stepData.error);
        clearInterval(agentIntervalId);
        return;
      }
      renderBoard(stepData.board_info);
      actionCount++;
      updateActionCount();

      if (stepData.done) {
        clearInterval(agentIntervalId);
        if (stepData.board_info.status === "won") {
          showMessage("¡QLearning clásico ha ganado!");
        } else {
          showMessage("QLearning clásico ha perdido...");
        }
      }
    }, 500);
  } catch (err) {
    console.error("Error al iniciar partida Q-learning clásico:", err);
  }
}


// ========== FUNCIONES COMUNES ==========

function renderBoard(gameState) {
  if (!gameState || gameState.error) {
    if (gameState?.error) {
      showMessage(gameState.error);
    }
    return;
  }

  const container = document.getElementById("boardContainer");
  container.innerHTML = ""; // Limpiar tablero anterior

  const { board, position, status } = gameState;
  const [playerR, playerC] = position;

  for (let r = 0; r < board.length; r++) {
    const rowDiv = document.createElement("div");
    rowDiv.classList.add("row");

    for (let c = 0; c < board[r].length; c++) {
      const cellDiv = document.createElement("div");
      cellDiv.classList.add("cell");

      // Asignar clases según el contenido
      if (r === 0 && c === 0) {
        cellDiv.classList.add("start");
      }
      if (board[r][c] === "X") {
        cellDiv.classList.add("forbidden");
      }
      if (board[r][c] === "F") {
        cellDiv.classList.add("finish");
      }

      // Resaltar posición del jugador si el juego no está perdido
      if (r === playerR && c === playerC && status !== "lost") {
        cellDiv.classList.add("player");
      }

      rowDiv.appendChild(cellDiv);
    }
    container.appendChild(rowDiv);
  }
}

function showMessage(msg) {
  const messageEl = document.getElementById("message");
  messageEl.textContent = msg;
}

function updateActionCount() {
  const counterEl = document.getElementById("actionCounter");
  counterEl.textContent = actionCount;
}
