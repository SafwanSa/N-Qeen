import numpy as np
import time
import random as rn
import copy

class Row:
    def __init__(self, row, queen, empty_place):
        self.row = row
        self.queen = queen
        self.empty_place = empty_place


class Solver:
    def __init__(self, n, mrv=False, lcv=False, mcv=False, fc=False, arc=False):
        self.n = n
        self.board = []
        self.rows = self.init()
        self.mrv = mrv
        self.lcv = lcv
        self.mcv = mcv
        self.fc = fc
        self.arc = arc
        self.counter = 0

    def get_min_val(self):
        min = 1000
        index = -1
        for i in range(len(self.rows)):
            if len(self.rows[i].empty_place) < min and self.rows[i].queen == -1:
                min = len(self.rows[i].empty_place)
                index = i

        return index

    def get_max_val(self):
        max = 0
        index = -1
        for i in range(len(self.rows)):
            if len(self.rows[i].empty_place) > max and self.rows[i].queen == -1:
                max = len(self.rows[i].empty_place)
                index = i

        return index

    def init(self):
        board = []
        for i in range(self.n):
            board.append(Row(i, -1, []))
        return board

    def solve_with_lcv(self, old_row):
        least = 0
        sum_places = 0
        index = 0
        row = copy.deepcopy(old_row)
        for i in range(len(row.empty_place)):
            self.board.append(row.empty_place[i][1])
            self.solve_with_variables()
            sum_places = sum(len(r.empty_place) for r in self.rows)
            if sum_places > least:
                self.board.pop()
                self.solve_with_variables()
                least = sum_places
                index = i
                temp = row.empty_place[i]
                row.empty_place.remove(temp)
                row.empty_place.insert(0, temp)
            else:
                self.board.pop()
                self.solve_with_variables()
        return index

    def solve_with_variables(self):
        for r in self.rows:
            for col in range(self.n):
                self.board.append(col)
                if self.is_valid_move():
                    if (r.row, col) not in r.empty_place:
                        r.empty_place.append((r.row, col))
                else:
                    if (r.row, col) in r.empty_place:
                        r.empty_place.remove((r.row, col))
                self.board.pop()

    def solve_with_mrv(self):
        self.solve_with_variables()
        next_row = self.rows[self.get_min_val()]
        return next_row

    def solve_with_mcv(self):
        self.solve_with_variables()
        next_row = self.rows[self.get_max_val()]
        return next_row

    def solve(self):
        self.solveUntil(self.n, self.rows[0])

    # Solve the problem with all required algorithms
    def solveUntil(self, n, row):
        self.counter+=1
        if len(self.board) == n:
            return True
        else:
            if row.row + 1 >= self.n:
                next_row = self.rows[row.row]
            else:
                next_row = self.rows[row.row + 1]

            # Backtracking with MCV
            if self.mcv:
                next_row = self.solve_with_mcv()

            # Backtracking with MRV
            if self.mrv:
                next_row = self.solve_with_mrv()

            # Backtracking with LCV
            index = -1
            if self.lcv:
                index = self.solve_with_lcv(row)
                self.board.append(index)
                row.queen = index
                if self.is_valid_move():
                    if self.solveUntil(n, next_row):
                        return True
                row.queen = -1
                self.board.pop()

            if self.fc:
                self.solve_with_variables()
                for _, col in row.empty_place:
                    index = col
                    self.board.append(col)
                    row.queen = col
                    if self.is_valid_move():
                        if self.solveUntil(n, next_row):
                            return True
                    self.board.pop()
                    row.queen = -1
                    self.solve_with_variables()

            if self.arc:
                for col in range(self.n):
                    self.board.append(col)
                    if self.is_valid_move():
                        self.solve_with_variables()
                        if len(next_row.empty_place) == 0:
                            self.board.pop()
                            continue
                        else:
                            if self.solveUntil(n, next_row):
                                return True
                    self.board.pop()


            # Normal backtracking
            for col in range(n):
                if col != index:
                    self.board.append(col)
                    row.queen = col
                    if self.is_valid_move():
                        if self.solveUntil(n, next_row):
                             return True
                    row.queen = -1
                    self.board.pop()
                else:
                    continue
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

    def solve_with_fc(self):
        pass


s = time.time()
solver = Solver(8, arc=True, mrv=True)
solver.solve()
e = time.time()
print(e - s)
solver.display()
print(solver.board)
print(solver.counter)
