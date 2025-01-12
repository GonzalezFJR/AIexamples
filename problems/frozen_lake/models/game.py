'''
Modelado del juego FrozenLake
'''

import random
import math

class FrozenLakeGame:
    def __init__(self, n=5, d=4, seed=None):
        """
        n: tamaño del tablero NxN
        d: número de casillas prohibidas (huecos)
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.status = "ongoing"   # 'ongoing', 'lost', 'won'
        self.position = [0, 0]    # (fila, columna)
        self.board = self._create_board()

        # Máximo de movimientos permitidos
        self.max_steps = n*n#(n * (n - 1)) // 2
        self.current_step = 0

        # Distancia máxima desde (0,0) a (n-1,n-1)
        self.dmax = math.sqrt((n-1)**2 + (n-1)**2)

    def _create_board(self):
        board = [["" for _ in range(self.n)] for _ in range(self.n)]
        # Casilla de inicio S
        board[0][0] = "S"
        # Casilla final F
        board[self.n-1][self.n-1] = "F"

        # Generar aleatoriamente d casillas prohibidas (X)
        valid_positions = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if not (r == 0 and c == 0) and not (r == self.n-1 and c == self.n-1)
        ]
        # set seed
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(valid_positions)
        for (r, c) in valid_positions[:self.d]:
            board[r][c] = "X"

        return board
    
    def reset(self):
        """
        Reinicia el juego (para un nuevo episodio).
        """
        self.status = "ongoing"
        self.position = [0, 0]
        self.board = self._create_board()
        self.current_step = 0
        return self.get_state()  # Devuelve la representación de estado

    def get_state(self):
        """
        Retornamos un entero que representa la posición del jugador:
           state = r * n + c
        """
        r, c = self.position
        return r * self.n + c

    def move(self, direction: str):
        """
        Mueve al jugador. Modifica self.status según la celda pisada.
        """
        if self.status != "ongoing":
            return

        r, c = self.position

        if direction == "up" and r > 0:
            r -= 1
        elif direction == "down" and r < self.n - 1:
            r += 1
        elif direction == "left" and c > 0:
            c -= 1
        elif direction == "right" and c < self.n - 1:
            c += 1

        self.position = [r, c]
        cell = self.board[r][c]

        if cell == "X":
            self.status = "lost"
        elif cell == "F":
            self.status = "won"
        else:
            self.status = "ongoing"


    def step(self, action):
        """
        Recibe 'action' como entero:
          0 = up, 1 = right, 2 = down, 3 = left
        Devuelve (next_state, reward, done, info).
        """
        if self.status != "ongoing":
            # Si ya terminó la partida, devolvemos sin cambios:
            return self.get_state(), 0.0, True, {}

        self.current_step += 1

        # Guardar la posición anterior
        r, c = self.position

        # Intentamos movernos
        new_r, new_c = r, c
        if action == 0 and r > 0:          # up
            new_r = r - 1
        elif action == 1 and c < self.n-1: # right
            new_c = c + 1
        elif action == 2 and r < self.n-1: # down
            new_r = r + 1
        elif action == 3 and c > 0:        # left
            new_c = c - 1
        else:
            # Movimiento inválido (fuera del tablero)
            # Penalización y no cambiamos posición
            reward = -0.2
            done = False
            if self.current_step >= self.max_steps:
                # Si además llegamos al límite de pasos
                self.status = "lost"
                reward = -1.0
                done = True
            return self.get_state(), reward, done, {}

        # Actualizamos la posición
        self.position = [new_r, new_c]
        cell = self.board[new_r][new_c]

        # Calculamos recompensas
        reward = 0.0
        done = False

        if cell == "X":
            reward = -1.0
            self.status = "lost"
            done = True
        elif cell == "F":
            reward = 2.0  # ganar => +2
            self.status = "won"
            done = True
        else:
            # Casilla vacía
            reward = -0.1
            done = False

        # Verificamos si se pasó el límite de pasos
        if self.current_step >= self.max_steps and not done:
            dist = math.sqrt((new_r - (self.n - 1))**2 + (new_c - (self.n - 1))**2)
            reward = 2.0 * (1.0 - (dist / self.dmax))
            self.status = "lost"
            done = True

        return self.get_state(), reward, done, {}

    def to_dict(self):
        """ Exportar el estado del juego a un diccionario """
        return {"board": self.board, "position": self.position, "status": self.status, "n": self.n, "d": self.d}