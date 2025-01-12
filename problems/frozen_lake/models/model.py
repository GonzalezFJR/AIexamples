'''
Script de python con modelos de agentes que juegan FrozenLake
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


########################################################################################################
### Modelo de Q-learning clásico

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.001):
        """
        n_states: número total de estados (por ejemplo, n*n si representas (r,c) como un entero).
        n_actions: número de acciones posibles (4 en FrozenLake: up, right, down, left).
        alpha: tasa de aprendizaje (lr).
        gamma: factor de descuento.
        epsilon_*: parámetros de exploración (epsilon-greedy).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Creamos la Q-table inicializada a 0
        self.Q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state):
        """
        Selecciona una acción con epsilon-greedy.
        state: entero que representa el estado actual.
        Retorna el índice de la acción [0..n_actions-1].
        """
        if random.random() < self.epsilon:
            # Acción aleatoria
            action = random.randint(0, self.n_actions - 1)
        else:
            # Acción greedy con respecto a la Q-table
            action = np.argmax(self.Q_table[state, :])
        return action

    def update(self, state, action, reward, next_state):
        """
        Actualiza la Q-table usando la ecuación de Q-learning:
          Q(s,a) <- (1-alpha)*Q(s,a) + alpha*(r + gamma * max_a' Q(s',a'))
        """
        best_next_action = np.argmax(self.Q_table[next_state, :])
        td_target = reward + self.gamma * self.Q_table[next_state, best_next_action]
        td_error = td_target - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self, episode):
        """
        Decaimiento exponencial de epsilon en función del episodio.
        Por ejemplo: epsilon = max(epsilon_min, exp(-decay*episode))
        """
        self.epsilon = max(self.epsilon_min, np.exp(-self.epsilon_decay * episode))

    def save(self, path="q_table.npy"):
        ''' Guarda la Q-table en un archivo .npy '''
        np.save(path, self.Q_table)

    def load(self, path="q_table.npy"):
        ''' Carga la Q-table desde un archivo .npy '''
        self.Q_table = np.load(path)



########################################################################################################
### Modelo de deep Q-learning (DQN) para FrozenLake
### Con este modelo se puede entrenar un agente para que juegue FrozenLake para un tablero definido
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        # state_size ~ n*n (ej. 16 si n=4)
        # action_size = 4 (arriba, derecha, abajo, izquierda)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc5 = nn.Linear(hidden_size//4, hidden_size//8)
        self.fc6 = nn.Linear(hidden_size//8, action_size)
        
    def forward(self, x):
        # x tendrá forma [batch_size, state_size]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x



########################################################################################################
### Modelo genérico de red neuronal para DQN
### Este modelo genérico juega a FrozenLake para *cualquier* tablero NxN
class GenericDQN(nn.Module):
    """
    Red neuronal para un tablero NxN con codificación one-hot de 4 canales
    Son 4 canales porque hay 4 posibles valores en cada celda (H, G, F, S)
    H = Hole (huecos), G = Goal (meta), F = Frozen (casillas por donde puedes ir), S = Start (posicion actual)
    => Tamaño de entrada: N*N*4 (aplanado)
    => Número de acciones: 4 (up, right, down, left)
    Aquí implementamos 4 capas ocultas.
    """
    def __init__(self, input_size=100, num_actions=4, hidden_sizes=[128,128,128,128]):
        super(GenericDQN, self).__init__()

        # Ejemplo de 4 capas ocultas
        # hidden_sizes puede ser una lista con las dimensiones de cada capa
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h
        self.hidden_layers = nn.ModuleList(layers)
        
        # Capa de salida
        self.output_layer = nn.Linear(in_dim, num_actions)

    def forward(self, x):
        """
        x: [batch_size, input_size] (ej. 100 para 5x5 con 4 canales)
        """
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        out = self.output_layer(x)
        return out