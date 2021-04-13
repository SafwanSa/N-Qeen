import numpy as np


class Position:
    def __init__(self, row, col):
        self.col = col
        self.row = row


class Solver:
    def __init__(self, n):
        self.n = n
        self.positions = np.full(fill_value=-1, shape=self.n, dtype=int).tolist()

    def solveNQeenOneSolution(self):
        hasSolution = self.solveNQeenSolutionUtil(self.n, 0, self.positions)

    def solveNQeenSolutionUtil(self, n, row, positions):
        if n == row:
            return True
        else:
            for col in range(n):
                foundSafe = True
                for queen in range(row):
                    if positions[queen].col == col or positions[queen].row - \
                            positions[queen].col == row - col or \
                            positions[queen].row + positions[queen].col == row + col:
                        foundSafe = False
                        break
                if foundSafe:
                    positions[row] = Position(row, col)
                    if self.solveNQeenSolutionUtil(n, row + 1, positions):
                        return True
            return False

    def to_board(self, positions):
        z = np.zeros((self.n, self.n))
        for pos in positions:
            z[pos.row, pos.col] = 1
        return z

    def display(self):
        print(self.to_board(self.positions))


solver = Solver(8)
solver.solveNQeenOneSolution()
solver.display()
