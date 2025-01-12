'''
Aplica el modelo entrenado para jugar una partida del juego Frozen Lake.
'''

import torch
import time
import numpy as np

from game import FrozenLakeGame
from model import DQN

N = 5
D = 4

def one_hot_state(state_int, state_size=N*N):
    v = np.zeros(state_size, dtype=np.float32)
    v[state_int] = 1.0
    return v

def play_game(n=N, d=D):
    # Cargar modelo
    state_size = n * n
    action_size = 4
    policy_net = DQN(state_size, action_size)
    policy_net.load_state_dict(torch.load("dqn.pth"))
    policy_net.eval()

    env = FrozenLakeGame(n, d)
    state_int = env.reset()
    done = False

    while not done:
        # Mostrar tablero en consola (o llamar a un endpoint para front)
        print(env.to_dict())  
        time.sleep(0.5)  # delay de 0.5s

        state = one_hot_state(state_int, state_size)
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0)  # [1, state_size]
            q_values = policy_net(state_t)
            action = q_values.argmax(dim=1).item()

        next_state_int, reward, done, _ = env.step(action)
        state_int = next_state_int

    # Mostrar el estado final
    print(env.to_dict())
    if env.status == "won":
        print("¡El agente ha ganado!")
    else:
        print("El agente ha perdido...")

if __name__ == "__main__":
    play_game()
