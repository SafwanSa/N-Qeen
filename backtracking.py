import random as rn

class Board:
    def __init__(self, N):
        self.N = N

    def initialize_random_board(self):
        board = np.zeros((self.N, self.N))
        while board.sum() < self.N:
            x = -1
            y = -1
            while board[x, y] == 1:
                x = rn.randint(0, self.N - 1)
                y = rn.randint(0, self.N - 1)
            board[x, y] = 1
        return board.tolist()