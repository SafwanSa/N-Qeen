import numpy as np
import time
import random as rn


class Row:
    def __init__(self, row, queen, empty_place):
        self.row = row
        self.queen = queen
        self.empty_place = empty_place


class Solver:
    def __init__(self, n, mrv=False, lcv=False):
        self.n = n
        self.board = []
        self.rows = self.init()
        self.mrv = mrv
        self.lcv = lcv

    def get_min_val(self):
        min = 1000
        index = -1
        for i in range(len(self.rows)):
            if len(self.rows[i].empty_place) < min and self.rows[i].queen == -1:
                min = len(self.rows[i].empty_place)
                index = i

        return index

    def init(self):
        board = []
        for i in range(self.n):
            board.append(Row(i, -1, []))
        return board

    # def solveNQeenOneSolution(self):
    #     self.solveNQeenSolutionUtil(self.n, 0)
    #
    # def solveNQeenSolutionUtil(self, n, row):
    #     if n == row:
    #         return True
    #     else:
    #         for col in range(n):
    #             self.board.append(col)
    #             if self.is_valid_move():
    #                 if self.solveNQeenSolutionUtil(n, row + 1):
    #                     return True
    #             self.board.pop()
    #         return False

    def solve_with_lcv(self, row):
        least = 0
        index = 0
        for col in range(self.n):
            sum = 0
            self.board.append(col)
            if self.is_valid_move():
                for r in self.rows:
                    if r != row:
                        r.empty_place = []
                        for c in range(self.n):
                            self.board.append(c)
                            if self.is_valid_move():
                                if (r.row, c) not in r.empty_place:
                                    r.empty_place.append((r.row, c))
                            else:
                                if (r.row, c) in r.empty_place:
                                    r.empty_place.remove((r.row, c))
                            self.board.pop()
                        sum += len(r.empty_place)
                if sum > least:
                    least = sum
                    index = col
            self.board.pop()
            return index

    def solve(self):
        self.solveUntil(self.n, self.rows[0])

    # Solve the problem with all required algorithms
    def solveUntil(self, n, row):
        if len(self.board) == n:
            return True
        else:
            if row.row + 1 >= self.n:
                next_row = self.rows[row.row]
            else:
                next_row = self.rows[row.row + 1]

            # Backtracking with MRV
            if self.mrv:
                for r in self.rows:
                    for col in range(n):
                        self.board.append(col)
                        if self.is_valid_move():
                            if (r.row, col) not in r.empty_place:
                                r.empty_place.append((r.row, col))
                        else:
                            if (r.row, col) in r.empty_place:
                                r.empty_place.remove((r.row, col))
                        self.board.pop()
                next_row = self.rows[self.get_min_val()]
                for rr in self.rows:
                    rr.empty_place.clear()

            # Backtracking with LCV
            if self.lcv:
                index = self.solve_with_lcv(row)
                self.board.append(index)
                row.queen = index
                if self.is_valid_move():
                    if self.solveUntil(n, next_row):
                        return True
                row.queen = -1
                self.board.pop()





            # Normal backtracking
            for col in range(n):
                self.board.append(col)
                row.queen = col
                if self.is_valid_move():
                    if self.solveUntil(n, next_row):
                        return True
                row.queen = -1
                self.board.pop()
            return False

    def is_valid_move(self):
        col = len(self.board) - 1
        for i in range(len(self.board) - 1):
            diff = abs(self.board[i] - self.board[col])
            if diff == 0 or diff == col - i:
                return False
        return True

    def to_board(self):
        board = np.zeros((self.n, self.n)).tolist()
        for i in range(self.n):
            try:
                board[self.board[i] - 1][i] = 1
            except IndexError:
                print(self.board)
        return board

    def display(self):
        for i in range(self.n):
            s = ''
            for j in range(self.n):
                if j == self.board[i]:
                    s += f'{1}  '
                else:
                    s += f'{0}  '
            print(s)

    def is_on_board(self, x, y):
        try:
            self.board[x]
            return True
        except IndexError:
            return False

    def get_possible_moves(self, x_coord, y_coord):
        moves = []
        for x, y in [(x_coord + i, y_coord + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0]:
            if self.is_on_board(x, y):
                moves.append((x, y))
        return moves


solver = Solver(8, mrv=False, lcv=True)
solver.solve()
solver.display()
print(solver.board)
