'''
Entrena un modelo de deep Q-learning (DQN) para FrozenLake, para un tablero de NxN dado.
'''

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

from models.game import FrozenLakeGame
from models.model import DQN

# Hiperparámetros (puedes ajustarlos a gusto)
N = 5
D = 4
STATE_SIZE = N * N
ACTION_SIZE = 4  # up, right, down, left
HIDDEN_SIZE = 128
LR = 5e-3
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99
MEMORY_SIZE = 1000

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        # transition = (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def one_hot_state(state_int, state_size=STATE_SIZE):
    """Convierte un entero (posición) en vector one-hot de tamaño state_size."""
    v = np.zeros(state_size, dtype=np.float32)
    v[state_int] = 1.0
    return v

def train(Ngames=1000, seed=None):
    env = FrozenLakeGame(N, D, seed=seed)
    policy_net = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_net = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    points = []

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    for episode in range(Ngames):
        state_int = env.reset()       # reset del entorno
        state = one_hot_state(state_int, STATE_SIZE)

        done = False
        episode_reward = 0.0

        while not done:
            # Selección de acción (epsilon-greedy)
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE - 1)
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax(dim=1).item()

            next_state_int, reward, done, _ = env.step(action)
            next_state = one_hot_state(next_state_int, STATE_SIZE)

            # Almacenar en el replay buffer
            memory.push((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            # Entrenamiento (sampling)
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state_t = torch.from_numpy(np.array(batch_state)).float()
                batch_action_t = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward_t = torch.FloatTensor(batch_reward).unsqueeze(1)
                batch_next_state_t = torch.from_numpy(np.array(batch_next_state)).float()
                batch_done_t = torch.BoolTensor(batch_done).unsqueeze(1)

                # Q(s,a)
                q_values = policy_net(batch_state_t).gather(1, batch_action_t)

                # Q target = r + gamma * max_a' Q_target(s',a') (si not done)
                with torch.no_grad():
                    max_q_next = target_net(batch_next_state_t).max(1)[0].unsqueeze(1)
                    q_target = batch_reward_t + GAMMA * max_q_next * (~batch_done_t)

                loss = F.mse_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Decaimiento de epsilon
            epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Sincronizar la red target cada X episodios
        if episode % 4 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"[Episode {episode}] Reward: {episode_reward:.2f}  Eps: {epsilon:.3f}")
        points.append(episode_reward)

    # Guardar pesos
    torch.save(policy_net.state_dict(), "dqn.pth")
    print("Entrenamiento finalizado, modelo guardado en dqn.pth")
    return points


########################################################################################################################
### Entrenar un modelo clásico

from models.model import QLearningAgent

def train_classic(n_episodes=2000, seed=None):
    # Parámetros
    N = 5
    D = 4
    max_steps = int(N*(N-1)/2)  # máximo de movimientos por episodio

    # Crear entorno
    env = FrozenLakeGame(n=N, d=D, seed=seed)
    n_states = N * N       # Si tu get_state() codifica (r,c) en [0..N*N-1]
    n_actions = 4          # up, right, down, left

    # Crear agente Q-learning
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,        # lr
        gamma=0.99,       
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.001
    )

    # Almacenar recompensas por episodio (opcional)
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()  # estado entero
        done = False
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Actualiza Q
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        # Decaimiento de epsilon según el episodio
        agent.decay_epsilon(episode)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}/{n_episodes}, Avg. Reward (últimos 100) = {avg_reward:.2f}")

    # Guardar la Q-table entrenada
    agent.save("q_table.npy")
    print("Entrenamiento clásico terminado. Q-table guardada en q_table.npy")
    return rewards_per_episode


########################################################################################
### Por defecto, entrenar un modelo DQN

import matplotlib.pyplot as plt
if __name__ == "__main__":
    #default = "DQL"
    default = "classic"

    points = []
    if default == "classic":
        rewards = train_classic()
        for i in range(0, len(rewards), 100):
            points.append(np.mean(rewards[i:i+100]))

    else:
        points = train()
        points = [sum(points[i:i+100]) / 100 for i in range(0, len(points), 100)]

    # Plot rewards
    plt.plot(points)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa promedio")
    plt.show()
    # save plot as image
    plt.savefig(os.path.join('static/', "training.png" if default!="classic" else "training_classic.png"))