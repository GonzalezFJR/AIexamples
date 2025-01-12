'''
Entrena un modelo de deep Q-learning (DQN) para FrozenLake, para **cualquier tablero** NxN.
'''

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from model import GenericDQN

# --- Parámetros del entorno ---
N = 5        # Tamaño del tablero NxN
D = 4        # Número de casillas prohibidas

# --- Parámetros de RL ---
NUM_ACTIONS = 4   # up=0, right=1, down=2, left=3
MAX_STEPS = (N * (N - 1)) // 2    # Límite de pasos por episodio
N_EPISODES = 500000                 # Número de episodios de entrenamiento
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99998
MEMORY_SIZE = 5000
TARGET_UPDATE = 32  # cada cuántos episodios sincronizamos la red target

# ==================================================
# 1) Definir entorno genérico con recompensas
# ==================================================
class GenericFrozenLake:
    """
    Cada episodio, generamos un tablero NxN con:
      - (0,0) casilla inicial (S).
      - (N-1,N-1) casilla final (F).
      - D casillas prohibidas al azar.
    El agente empieza en (0,0). 
    Recompensas (ejemplo):
      - -1 si pisa una casilla prohibida (X) => done
      - +1 si llega a la meta => done
      - 0 si movimiento normal
      - -0.2 si intenta salir del tablero
      - Al agotar MAX_STEPS sin terminar, done y reward=-1
    """
    def __init__(self, n=5, d=4):
        self.n = n
        self.d = d
        self.max_steps = (n * (n-1)) // 2
        self.reset()

    def reset(self):
        # Crea el tablero
        self.board = [["" for _ in range(self.n)] for _ in range(self.n)]
        # Marcar inicio y final
        self.board[0][0] = "S"
        self.board[self.n-1][self.n-1] = "F"

        # Generar D casillas prohibidas aleatorias
        valid_positions = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if (r, c) not in [(0,0), (self.n-1, self.n-1)]
        ]
        random.shuffle(valid_positions)
        for (r, c) in valid_positions[:self.d]:
            self.board[r][c] = "X"

        self.player_pos = [0, 0]
        self.current_step = 0
        self.done = False

        return self.get_observation()

    def get_observation(self):
        """
        Devuelve un vector aplanado de longitud n*n*4 con one-hot por celda:
          - [1,0,0,0] => libre
          - [0,1,0,0] => posición jugador
          - [0,0,1,0] => final
          - [0,0,0,1] => prohibida
        """
        # Construimos un array [n, n, 4]
        obs = np.zeros((self.n, self.n, 4), dtype=np.float32)
        
        # Rellenar
        for r in range(self.n):
            for c in range(self.n):
                val = self.board[r][c]
                if val == "X":
                    obs[r, c, 3] = 1.0  # prohibida
                elif val == "F":
                    obs[r, c, 2] = 1.0  # final
                else:
                    obs[r, c, 0] = 1.0  # libre

        # Marcar la posición del jugador
        pr, pc = self.player_pos
        # Notar que si la posición del jugador coincide con la final,
        # en la práctica se prioriza la marca "jugador" (o se hace un merge).
        # Para la demo, sobrescribimos a la 'player' en canal 1
        obs[pr, pc, :] = 0.0
        obs[pr, pc, 1] = 1.0

        # Aplanar
        return obs.flatten()  # shape => (n*n*4, )

    def step(self, action):
        """
        Recibe action in {0,1,2,3} = {up, right, down, left}.
        Devuelve (obs, reward, done, info).
        """
        if self.done:
            return self.get_observation(), 0.0, True, {}

        self.current_step += 1
        r, c = self.player_pos
        nr, nc = r, c

        # Movimiento
        if action == 0 and r > 0:          # up
            nr = r - 1
        elif action == 1 and c < self.n-1: # right
            nc = c + 1
        elif action == 2 and r < self.n-1: # down
            nr = r + 1
        elif action == 3 and c > 0:        # left
            nc = c - 1
        else:
            # Movimiento fuera del tablero => penalización
            reward = -0.2
            done = False
            if self.current_step >= self.max_steps:
                reward = -1.0
                done = True
                self.done = True
            return self.get_observation(), reward, done, {}

        # Actualizar pos
        self.player_pos = [nr, nc]
        cell = self.board[nr][nc]

        reward = 0.0
        done = False
        if cell == "X":
            reward = -1.0
            done = True
        elif cell == "F":
            reward = 1.0
            done = True
        else:
            # Casilla normal => 0
            reward = 0.0

        if self.current_step >= self.max_steps and not done:
            # Se agotan pasos => -1
            reward = -1.0
            done = True

        self.done = done
        return self.get_observation(), reward, done, {}

# ==================================================
# 2) Replay Buffer
# ==================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*minibatch)
        return np.array(s), a, r, np.array(s2), d

    def __len__(self):
        return len(self.buffer)

# ==================================================
# 3) Funciones auxiliares
# ==================================================
def select_action(state, policy_net, epsilon):
    """
    Epsilon-greedy.
    state: numpy array shape (100,)
    """
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0)  # [1,100]
            q_values = policy_net(state_t)
            action = q_values.argmax(dim=1).item()
        return action

# ==================================================
# 4) Bucle de entrenamiento
# ==================================================
def train_dqn():
    env = GenericFrozenLake(N, D)
    input_size = N*N*4  # 5*5*4 = 100
    policy_net = GenericDQN(input_size=input_size, num_actions=NUM_ACTIONS, hidden_sizes=[256,128,64,32])
    target_net = GenericDQN(input_size=input_size, num_actions=NUM_ACTIONS, hidden_sizes=[256,128,64,32])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START
    points = []

    for episode in range(N_EPISODES):
        state = env.reset()  # shape=(100,)
        done = False
        episode_reward = 0.0

        while not done:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Entrenamos si tenemos suficiente en memoria
            if len(memory) >= BATCH_SIZE:
                batch_s, batch_a, batch_r, batch_s2, batch_d = memory.sample(BATCH_SIZE)

                # Convertir a tensores
                batch_s_t = torch.from_numpy(batch_s).float()      # [B, 100]
                batch_a_t = torch.LongTensor(batch_a).unsqueeze(1) # [B,1]
                batch_r_t = torch.FloatTensor(batch_r).unsqueeze(1)# [B,1]
                batch_s2_t = torch.from_numpy(batch_s2).float()     # [B, 100]
                batch_d_t = torch.BoolTensor(batch_d).unsqueeze(1)  # [B,1]

                # Q(s,a)
                q_values = policy_net(batch_s_t).gather(1, batch_a_t)  # [B,1]

                with torch.no_grad():
                    # Max Q(s2, a2) con la target net
                    q_next = target_net(batch_s2_t).max(dim=1)[0].unsqueeze(1)  # [B,1]
                    # Si done, no sumamos gamma*q_next
                    q_target = batch_r_t + GAMMA * q_next * (~batch_d_t)

                loss = F.mse_loss(q_values, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decaimiento de epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Sincronizar target net cada X episodios
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        points.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(points[-100:])
            print(f"Episode {episode} | Reward: {avg_reward:.2f} | Eps: {epsilon:.3f}")

    # Guardar modelo
    torch.save(policy_net.state_dict(), "dqn_generic.pth")
    print("Entrenamiento finalizado. Modelo guardado en dqn_generic.pth")
    return points

# ==================================================
# 5) Punto de entrada
# ==================================================
if __name__ == "__main__":
    points = train_dqn()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 8))

    # Media cada 100 episodios
    means = [np.mean(points[i:i+100]) for i in range(0, len(points), 100)]
    ax.plot(means, label="Reward (media móvil 100)")

    ax.set_title("Entrenamiento DQN genérico")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward")

    # save as image
    plt.savefig("dqn_generic_training.png")
