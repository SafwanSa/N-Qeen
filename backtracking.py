import random as rn
import numpy as np

class Board:
    def __init__(self, N):
        self.N = N
        self.board = np.zeros((self.N, self.N)).tolist()

    def display(self):
        print(np.array(self.board))

    def solve(self):
        self.solve_backtracking(0)

    def solve_backtracking(self, column):
        if np.array(self.board).sum() == self.N:
            return True
        else:
            for i in range(self.N):
                if self.is_valid_move(i, column):
                    self.board[i][column] = 1

                    if self.solve_backtracking(column + 1):
                        return True
                    self.board[i][column] = 0
            return False

    def is_valid_move(self, x, y):
        att = self.num_of_attacks_on_each(self.board, self.N, x, y)
        return att == 0

    def num_of_attacks_on_each(self, m, N, x, y):
        map = np.array(m)
        right_lower_diag = []
        right_upper_diag = []
        left_upper_diag = []
        left_lower_diag = []
        for i, j in zip(range(x + 1, N), range(y + 1, N)):
            right_lower_diag.append(map[i][j])

        for i, j in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
            left_upper_diag.append(map[i][j])

        for i, j in zip(range(x + 1, N), range(y - 1, -1, -1)):
            left_lower_diag.append(map[i][j])

        for i, j in zip(range(x - 1, -1, -1), range(y + 1, N)):
            right_upper_diag.append(map[i][j])

        row_attacks = map[x, 0:].sum()
        column_attacks = map[:, y].sum()
        upper_diag_attakcs = np.array(left_upper_diag).sum() + np.array(right_upper_diag).sum()
        lower_diag_attakcs = np.array(left_lower_diag).sum() + np.array(right_lower_diag).sum()
        attacks = row_attacks + column_attacks + upper_diag_attakcs + lower_diag_attakcs
        return attacks




board = Board(8)
board.solve()
board.display()