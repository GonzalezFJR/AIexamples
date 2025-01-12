# main.py

import os
import random
import numpy as np
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Importar funciones de entrenamiento
# Asumiendo que tu archivo se llama train.py y está en la carpeta "models"
from models import train

# Importar clases para el juego y agentes
from models.game import FrozenLakeGame
from models.model import DQN, QLearningAgent

app = FastAPI()

# Montar carpeta "static" para servir archivos estáticos (CSS, JS, imágenes...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
#    VARIABLES GLOBALES
# ===========================
# -- Para juego humano --
current_game: FrozenLakeGame = None

# -- Para DQN --
policy_net = None
agent_env: FrozenLakeGame = None
agent_done: bool = True
agent_state_int: int = None

N = 5
D = 4

# -- Hiperparámetros DQN (ajusta a gusto) --
STATE_SIZE = N*N
ACTION_SIZE = 4
HIDDEN_SIZE = 128

# -- Para Q-learning clásico --
q_agent: QLearningAgent = None
classic_done: bool = True
classic_state: int = None


# ===========================
#     FUNCIONES AUX
# ===========================
def one_hot_state(state_int, state_size=STATE_SIZE):
    """Convierte un entero (posición) en vector one-hot de dimensión state_size."""
    v = np.zeros(state_size, dtype=np.float32)
    v[state_int] = 1.0
    return v


# =======================================
#   RUTAS PARA SERVIR PÁGINAS ESTÁTICAS
# =======================================

@app.get("/", response_class=FileResponse)
def serve_landing():
    """
    Página principal (landing) con los 3 botones:
    1) Juego humano
    2) Q-learning clásico
    3) Deep Q-Learning
    """
    return FileResponse(os.path.join(STATIC_DIR, "landing.html"))

@app.get("/human", response_class=FileResponse)
def serve_human():
    """
    Página para jugar manualmente (humano).
    """
    return FileResponse(os.path.join(STATIC_DIR, "human.html"))

@app.get("/classic", response_class=FileResponse)
def serve_classic():
    """
    Página para Q-learning clásico.
    Carga la Q-table si aún no está en memoria.
    """
    global q_agent
    if q_agent is None:
        # Crear agente
        q_agent = QLearningAgent(n_states=N*N, n_actions=4)
        # Intentar cargar Q-table de /static
        q_table_path = os.path.join(STATIC_DIR, "q_table.npy")
        try:
            q_agent.load(q_table_path)
            print("Q-table cargada correctamente desde /static/q_table.npy.")
        except FileNotFoundError:
            print("No se encontró q_table.npy en /static. Debes entrenar con /train_classic.")

    return FileResponse(os.path.join(STATIC_DIR, "classic.html"))

@app.get("/dqn", response_class=FileResponse)
def serve_dqn():
    """
    Página para Deep Q-Learning.
    Carga el modelo DQN si aún no está en memoria.
    """
    global policy_net
    if policy_net is None:
        policy_net = DQN(STATE_SIZE, ACTION_SIZE, hidden_size=HIDDEN_SIZE)
        dqn_path = os.path.join(STATIC_DIR, "dqn.pth")
        try:
            policy_net.load_state_dict(torch.load(dqn_path))
            policy_net.eval()
            print("Modelo DQN cargado correctamente desde /static/dqn.pth.")
        except FileNotFoundError:
            print("No se encontró dqn.pth en /static. Debes entrenar con /train_dqn.")

    return FileResponse(os.path.join(STATIC_DIR, "dqn.html"))


# ===========================
#      RUTAS DE ENTRENAR
# ===========================
@app.get("/train_dqn")
def train_dqn(episodes: int = 1000):
    """
    Entrena un modelo DQN por 'episodes' episodios.
    - Genera un seed aleatorio para reproducibilidad
    - Llama a train.train(...) 
    - Mueve dqn.pth a /static/dqn.pth
    - Recarga a policy_net en memoria
    - Devuelve el seed y stats de entrenamiento
    """
    global policy_net

    # Generar semilla aleatoria
    seed = random.randint(0, 999999)
    print(f"Iniciando entrenamiento DQN con seed={seed}, episodes={episodes}")

    # Ejecutar entrenamiento (usa la función de train.py)
    rewards = train.train(Ngames=episodes, seed=seed)  # produce dqn.pth en la carpeta actual

    # Mover dqn.pth a /static/dqn.pth
    src_path = "dqn.pth"
    dst_path = os.path.join(STATIC_DIR, "dqn.pth")
    if os.path.exists(src_path):
        os.replace(src_path, dst_path)
        print("Modelo DQN movido a:", dst_path)

    # Recargar en memoria
    policy_net = DQN(STATE_SIZE, ACTION_SIZE, hidden_size=HIDDEN_SIZE)
    try:
        policy_net.load_state_dict(torch.load(dst_path))
        policy_net.eval()
        print("policy_net recargado en memoria.")
    except FileNotFoundError:
        return {"error": f"No se pudo encontrar {dst_path} después de entrenar."}
    
    return {
        "message": f"Entrenamiento DQN completado, episodes={episodes}, seed={seed}",
        "seed": seed,
        "rewards": rewards[-10:]  # últimas recompensas para debug
    }


@app.get("/train_classic")
def train_classic_q(episodes: int = 2000):
    """
    Entrena un modelo Q-learning clásico por 'episodes' episodios.
    - Genera un seed aleatorio
    - Llama a train.train_classic(...) 
    - Mueve q_table.npy a /static/q_table.npy
    - Recarga la q_table en q_agent
    - Devuelve el seed y stats de entrenamiento
    """
    global q_agent

    # Generar semilla aleatoria
    seed = random.randint(0, 999999)
    print(f"Iniciando entrenamiento clásico con seed={seed}, episodes={episodes}")

    # Ejecutar entrenamiento (usa la función de train.py)
    rewards = train.train_classic(n_episodes=episodes, seed=seed)  # produce q_table.npy en la carpeta actual

    # Mover q_table.npy a /static/q_table.npy
    src_path = "q_table.npy"
    dst_path = os.path.join(STATIC_DIR, "q_table.npy")
    if os.path.exists(src_path):
        os.replace(src_path, dst_path)
        print("Q-table movida a:", dst_path)

    # Volver a crear / recargar el agente
    q_agent = QLearningAgent(n_states=N*N, n_actions=4)
    try:
        q_agent.load(dst_path)
        print("q_agent recargado en memoria con la nueva Q-table.")
    except FileNotFoundError:
        return {"error": f"No se encontró {dst_path} después de entrenar."}

    return {
        "message": f"Entrenamiento clásico completado, episodes={episodes}, seed={seed}",
        "seed": seed,
        "rewards": rewards[-10:]  # últimas recompensas para debug
    }


# ===========================
#   PARTIDA MANUAL (HUMANO)
# ===========================
@app.get("/new_game")
def new_game(n: int = 5, d: int = 4):
    """
    Crea una nueva partida manual y devuelve el estado inicial.
    """
    global current_game
    current_game = FrozenLakeGame(n, d)
    return current_game.to_dict()

@app.post("/move")
def make_move(direction: str):
    """
    Mueve al jugador (usuario) en la dirección indicada.
    """
    global current_game
    if not current_game:
        return {"error": "No game in progress"}

    current_game.move(direction)
    return current_game.to_dict()

@app.get("/status")
def get_status():
    """
    Devuelve el estado actual de la partida manual.
    """
    if current_game:
        return current_game.to_dict()
    return {"error": "No game in progress"}


# ===========================
#      PARTIDA DQN
# ===========================
@app.get("/start_play")
def start_play(seed: int = None):
    """
    Inicia (resetea) una nueva partida para el agente DQN.
    Si 'seed' viene definido, se usará para crear el juego con esa semilla.
    """
    global agent_env, agent_done, agent_state_int, policy_net
    if policy_net is None:
        return {"error": "El modelo DQN no está cargado."}

    agent_env = FrozenLakeGame(N, D, seed=seed)
    agent_state_int = agent_env.reset()
    agent_done = False
    return agent_env.to_dict()

@app.get("/play_step")
def play_step():
    """
    Ejecuta un paso con el agente DQN y devuelve el estado tras la acción.
    """
    global agent_env, agent_done, agent_state_int, policy_net
    if policy_net is None:
        return {"error": "El modelo DQN no está cargado."}

    if agent_env is None:
        return {"error": "No hay partida DQN iniciada. Llama a /start_play primero."}

    if agent_done:
        return {"error": "Partida DQN finalizada. Vuelve a iniciar con /start_play."}

    # Construimos el one-hot a partir de agent_state_int
    state_vec = one_hot_state(agent_state_int, STATE_SIZE)
    state_t = torch.from_numpy(state_vec).unsqueeze(0)  # [1, state_size]
    
    with torch.no_grad():
        q_values = policy_net(state_t)
        action = q_values.argmax(dim=1).item()  # acción con mayor valor Q

    next_state_int, reward, done, _ = agent_env.step(action)
    agent_state_int = next_state_int
    agent_done = done

    return {
        "board_info": agent_env.to_dict(),
        "reward": reward,
        "done": done,
    }


# ===========================
# Q-LEARNING CLÁSICO
# ===========================
@app.get("/start_classic")
def start_classic(seed: int = None):
    """
    Inicia (resetea) una nueva partida para el agente Q-learning Clásico.
    Si 'seed' viene definido, se usará para crear el juego con esa semilla.
    """
    global agent_env, classic_done, classic_state, q_agent
    if q_agent is None:
        return {"error": "No se ha cargado la Q-table. Usa /train_classic o ingresa a /classic."}

    agent_env = FrozenLakeGame(n=N, d=D, seed=seed)
    classic_state = agent_env.reset()
    classic_done = False
    return agent_env.to_dict()

@app.get("/play_step_classic")
def play_step_classic():
    """
    Da un paso usando la política de la Q-table (greedy, sin exploración).
    """
    global agent_env, classic_done, classic_state, q_agent

    if q_agent is None or agent_env is None:
        return {"error": "No hay partida clásica iniciada. Llama a /start_classic primero."}

    if classic_done:
        return {"error": "La partida clásica ya finalizó. Llama a /start_classic."}

    # Acción greedy (argmax sobre la Q-table)
    action = np.argmax(q_agent.Q_table[classic_state, :])

    next_state, reward, done, _ = agent_env.step(action)
    classic_state = next_state
    classic_done = done

    return {
        "board_info": agent_env.to_dict(),
        "reward": reward,
        "done": done
    }


# =========================================
#   Ejecución con uvicorn (si se desea)
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
