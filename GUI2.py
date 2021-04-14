import sys
from PyQt5.QtWidgets import (QWidget, QToolTip,
                             QPushButton, QApplication, QLabel, QVBoxLayout, QLineEdit, QCheckBox, QRadioButton,
                             QHBoxLayout)
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot

import timeit
from queue import PriorityQueue
import numpy as np
import time
import random as rn
import copy
import pygame as pg
import itertools


# Genetic Algorithm solver
class GeneticSolver:
    def __init__(self, N, pop_size, mr, gen):
        self.N = N
        self.pop_size = pop_size
        self.mr = mr
        self.gen = gen

    def solve(self):
        start = timeit.default_timer()
        population = Population(self.mr, self.pop_size, self.N)
        solution = self.draw_board(self.N, population)
        stop = timeit.default_timer()
        return solution.generation, (stop - start)

    def draw_board(self, N, population):
        pg.init()
        square_length = 40
        board_width, board_height = N * square_length, N * square_length
        window = pg.display.set_mode((board_width, board_height))


        GREY = pg.Color('gray')
        WHITE = pg.Color('white')
        colors = itertools.cycle((WHITE, GREY))
        queen_img = pg.image.load("q.png")
        queen_img = pg.transform.scale(queen_img, (square_length, square_length))
        board = pg.Surface((board_width, board_height))



        game_exit = False
        while not game_exit:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    game_exit = True

            while not population.solution != None and population.generation <= self.gen:
                population.calc_fitness()

                population.natural_selection()

                population.generate()

                bb = population.get_min_attacks_and_best()[1]
                if population.solution != None:
                    bb = population.solution
                for y in range(0, board_height, square_length):
                    for x in range(0, board_width, square_length):
                        square = (x, y, square_length, square_length)
                        pg.draw.rect(board, next(colors), square)
                    if N % 2 == 0:
                        next(colors)
                    for i in range(N):
                        for j in range(N):
                            if bb.board[i][j] == 1:
                                board.blit(queen_img, (j * square_length, i * square_length))
                pg.display.update()
                window.blit(board, (0, 0))
                pg.display.flip()
                pg.time.delay(10)

                print("Generation: {}   Min Num of Attacks: {}".format(population.generation, population.get_min_attacks_and_best()[0]))
                if population.solution != None:
                    game_exit = True
        return population

# The part that makes the genetic population
class DNA:
    def __init__(self, N):
        self.N = N
        self.fitness = 0
        self.gene = []
        self.board = self.create_board()
        for i in range(N):
            self.gene.append(rn.randint(1, self.N))
        self.set_board_from_gene()

    def create_board(self):
        board = [[0 for i in range(self.N)] for j in range(self.N)]
        return board

    def set_board_from_gene(self):
        self.board = np.zeros((self.N, self.N)).tolist()
        for i in range(self.N):
            self.board[self.gene[i] - 1][i] = 1

    def set_gene_from_board(self):
        for j in range(self.N):
            for i in range(self.N):
                if self.board[i][j] == 1:
                    self.gene[j] = i + 1

    def get_num_of_attacks(self):
        hits = 0
        col = 0
        self.set_board_from_gene()
        for dna in self.gene:
            try:
                for i in range(col - 1, -1, -1):
                    if self.board[dna - 1][i] == 1:
                        hits += 1
            except IndexError:
                print(self.gene, "\n\n" ,np.array(self.board))
                quit()
            for i, j in zip(range(dna - 2, -1, -1), range(col - 1, -1, -1)):
                if self.board[i][j] == 1:
                    hits += 1
            for i, j in zip(range(dna, self.N, 1), range(col - 1, -1, -1)):
                if self.board[i][j] == 1:
                    hits += 1
            col += 1
        return hits

    def set_fitness(self):
        attacks = self.get_num_of_attacks()
        self.fitness = attacks

    def crossover(self, other):
        dna = DNA(self.N)
        mid = rn.randint(0, self.N)
        new_gene = self.gene[0:mid] + other.gene[mid:]
        dna.gene = new_gene
        dna.board = np.zeros((self.N, self.N)).tolist()
        dna.set_board_from_gene()
        dna.set_fitness()
        return dna

    def mutate(self, rate):
        for i in range(self.N):
            num = rn.random()
            if num < rate:
                self.gene[i] = rn.randint(1, self.N - 1)
        self.set_board_from_gene()

# A group of DNAs
class Population:
    def __init__(self, mutationRate, popmax, N):
        self.mutationRate = mutationRate
        self.popmax = popmax
        self.N = N
        self.generation = 0
        self.matingPool = []
        self.population = []
        self.solution = None
        for i in range(popmax):
            self.population.append(DNA(self.N))

    def calc_fitness(self):
        for dna in self.population:
            dna.set_fitness()

    def natural_selection(self):
        self.matingPool = []
        fitness = 0
        for dna in self.population:
            if dna.fitness != 0:
                fitness = 1 / dna.fitness
            else:
                self.solution = dna

            n = int(round(fitness * 10))
            for i in range(n):
                self.matingPool.append(copy.deepcopy(dna))

    def generate(self):
        new_pop = []
        if len(self.matingPool) != 0:
            for i in range(len(self.population)):
                a = rn.randint(0, len(self.matingPool) - 1)
                b = rn.randint(0, len(self.matingPool) - 1)
                dna_A = self.matingPool[a]
                dna_B = self.matingPool[b]

                new_dna = dna_A.crossover(dna_B)

                new_dna.mutate(self.mutationRate)

                new_pop.append(new_dna)
        self.population = copy.deepcopy(new_pop)
        self.generation += 1

    def get_min_attacks_and_best(self):
        min = 1000
        best = None
        for dna in self.population:
            if dna.fitness < min:
                min = dna.fitness
                best = dna
        return min, dna

# Constrains Satisfaction Proplem Variable
class Row:
    def __init__(self, row, queen, empty_place):
        self.row = row
        self.queen = queen
        self.empty_place = empty_place

# Backtracking Solver with all options MCV, MRV, LCV, FC, and ARC
class BacktrackingSolver:
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
        s = time.time()
        # Solve the problem
        self.solveUntil(self.n, self.rows[0])
        e = time.time()

        bb = self.to_board()

        print(self.board)
        print(np.array(bb))
        pg.init()
        square_length = 40
        board_width, board_height = self.n * square_length, self.n * square_length
        window = pg.display.set_mode((board_width, board_height))


        GREY = pg.Color('gray')
        WHITE = pg.Color('white')
        colors = itertools.cycle((WHITE, GREY))
        queen_img = pg.image.load("q.png")
        queen_img = pg.transform.scale(queen_img, (square_length, square_length))
        board = pg.Surface((board_width, board_height))

        game_exit = False
        while not game_exit:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    game_exit = True

                for y in range(0, board_height, square_length):
                    for x in range(0, board_width, square_length):
                        square = (x, y, square_length, square_length)
                        pg.draw.rect(board, next(colors), square)
                    if self.n % 2 == 0:
                        next(colors)
                    for i in range(self.n):
                        for j in range(self.n):
                            if bb[i][j] == 1:
                                board.blit(queen_img, (j * square_length, i * square_length))
                pg.display.update()
                window.blit(board, (0, 0))
                pg.display.flip()
                pg.time.delay(10)

        return e - s


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
                for _, col in row.empty_place:
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
                board[self.board[i]][i] = 1
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

# A* Solver
class AStarSolver:
    def __init__(self, Nn):
        self.Nn = Nn
        self.solution = None
        self.queue = PriorityQueue()
        self.closed_set = []

    def is_visited(self, closed_set, board):
        e = False
        for m in closed_set:
            if np.array_equal(m, board):
                e = True
        return e

    def solve(self):
        start = timeit.default_timer()
        steps = self.draw_board(self.Nn)
        stop = timeit.default_timer()
        return steps, (stop - start)

    def draw_board(self, N):
        solution = None
        finish = False
        board = Board(N, 0, 0, [])
        self.queue.put(board)
        counter = 0

        pg.init()
        square_length = 40
        board_width, board_height = N * square_length, N * square_length
        window = pg.display.set_mode((board_width, board_height))


        GREY = pg.Color('gray')
        WHITE = pg.Color('white')
        colors = itertools.cycle((WHITE, GREY))
        queen_img = pg.image.load("q.png")
        queen_img = pg.transform.scale(queen_img, (square_length, square_length))
        board_s = pg.Surface((board_width, board_height))



        game_exit = False
        while not game_exit:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    game_exit = True

            while not finish:
                if self.queue.empty():
                    print("No Solution Found!!")
                    quit()
                board = self.queue.get()
                for y in range(0, board_height, square_length):
                    for x in range(0, board_width, square_length):
                        square = (x, y, square_length, square_length)
                        pg.draw.rect(board_s, next(colors), square)
                    if N % 2 == 0:
                        next(colors)
                    for i in range(N):
                        for j in range(N):
                            if board.board[i][j] == 1:
                                board_s.blit(queen_img, (j * square_length, i * square_length))
                if board.hn == 0:
                    finish = True
                    solution = copy.deepcopy(board)
                else:
                    self.closed_set.append(np.array(board.board))
                    for i in range(N):
                        for j in range(N):
                            if board.board[i][j] == 1:
                                possible_moves = board.get_possible_moves(N, i, j)
                                for move in possible_moves:
                                    board.board[i][j] = 0
                                    board.board[move[0]][move[1]] = 1
                                    child = Board(N, board.gn, board.hn, copy.deepcopy(board.board))
                                    board.board[i][j] = 1
                                    board.board[move[0]][move[1]] = 0
                                    if not self.is_visited(self.closed_set, np.array(child.board)):
                                        self.queue.put(child)
                counter += 1
                pg.display.update()
                window.blit(board_s, (0, 0))
                pg.display.flip()
                pg.time.delay(10)

                print("Steps: {}   Num of Attacks: {}".format(counter, board.hn))
                if solution != None:
                    game_exit = True
        return counter

# The board used in A* Algorithm
class Board:
    def __init__(self, N, cost, parent_attacks, board):
        self.N = N
        self.board = self.initialize_random_board() if len(board) == 0 else board
        self.gn = self.get_cost(cost, parent_attacks) if cost != 0 else 0
        self.hn = self.calculate_heuristic()
        self.fn = self.gn + self.hn

    def __lt__(self, other):
        return self.fn < other.fn

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

    def get_num_of_attacks(self):
        each = 0
        for j in range(self.N):
            for i in range(self.N):
                if self.board[i][j] == 1:
                    each += self.num_of_attacks_on_each(np.array(self.board), self.N, i, j)
        return each / 2

    def get_cost(self, cost_from_root, parent_attacks):
        return self.N + (self.get_num_of_attacks() - parent_attacks) + cost_from_root

    def calculate_heuristic(self):
        return self.get_num_of_attacks()

    def num_of_attacks_on_each(self, map, N, x, y):
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

        row_attacks = map[x, 0:].sum() - 1
        column_attacks = map[:, y].sum() - 1
        upper_diag_attakcs = np.array(left_upper_diag).sum() + np.array(right_upper_diag).sum()
        lower_diag_attakcs = np.array(left_lower_diag).sum() + np.array(right_lower_diag).sum()
        attacks = row_attacks + column_attacks + upper_diag_attakcs + lower_diag_attakcs
        return attacks

    def is_on_board(self, x, y):
        try:
            self.board[int(x)][int(y)]
            if x < 0 or y < 0 or x >= self.N or y >= self.N:
                return False
            else:
                if self.board[x][y] == 1:
                    return False
                else:
                    return True


        except (ValueError, IndexError):
            return False
        else:
            return True

    def get_possible_moves(self, N, x_coord, y_coord):
        moves = []
        for x, y in [(x_coord + i, y_coord + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0]:
            if self.is_on_board(x, y):
                moves.append((x, y))
        return moves

# The part responsible of the GUI
class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainLayout = QVBoxLayout()

        greeting = QLabel("Welcome To Assignment 2\n Enter the number of queens", self)
        greeting.setAlignment(QtCore.Qt.AlignCenter)

        self.textbox = QLineEdit(self)

        start = QLabel("How do you want to solve it?", self)
        start.setAlignment(QtCore.Qt.AlignCenter)

        btn = QPushButton('A* Search', self)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.solve_with_a_star)

        btn2 = QPushButton('Genetic Algorithm', self)
        btn2.resize(btn2.sizeHint())
        btn2.clicked.connect(self.solve_with_genetic)

        btn3 = QPushButton('Backtracking...', self)
        btn3.resize(btn3.sizeHint())
        btn3.clicked.connect(self.solve_with_backtracking)

        self.mrv = QCheckBox("MRV")
        self.mcv = QCheckBox("MCV")
        self.lcv = QCheckBox("LCV")

        self.mainLayout.addWidget(greeting)
        self.mainLayout.addWidget(self.textbox)
        self.mainLayout.addWidget(start)
        self.mainLayout.addWidget(btn)
        self.mainLayout.addWidget(btn2)
        self.mainLayout.addWidget(btn3)

        self.hLayout = QHBoxLayout()

        self.hLayout.addWidget(self.mrv)
        self.hLayout.addWidget(self.mcv)
        self.hLayout.addWidget(self.lcv)

        self.mainLayout.addLayout(self.hLayout)

        self.hLayout2 = QHBoxLayout()
        self.none = QRadioButton("None")
        self.none.setChecked(True)
        self.none.country = "None"
        self.hLayout2.addWidget(self.none)
        self.fc = QRadioButton("FC")
        self.fc.country = "FC"
        self.hLayout2.addWidget(self.fc)
        self.arc = QRadioButton("ARC")
        self.arc.country = "ARC"
        self.hLayout2.addWidget(self.arc)

        self.mainLayout.addLayout(self.hLayout2)

        self.setLayout(self.mainLayout)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Solve N-Queen Problem')
        self.show()

    # Solving the problem with A*
    @pyqtSlot()
    def solve_with_a_star(self):
        n = 0
        try:
            n = int(self.textbox.text())
            stop = QPushButton('Close', self)
            stop.resize(stop.sizeHint())
            stop.clicked.connect(self.stop_solving)
            self.mainLayout.addWidget(stop)

            # start search
            aSolver = AStarSolver(n)
            steps, time = aSolver.solve()

            stepsW = QLabel("A* Steps: {}".format(steps), self)
            stepsW.setAlignment(QtCore.Qt.AlignCenter)

            timeW = QLabel("A* Time: {}".format(round(time)), self)
            timeW.setAlignment(QtCore.Qt.AlignCenter)

            self.mainLayout.addWidget(timeW)
            self.mainLayout.addWidget(stepsW)
        except:
            pass

    @pyqtSlot()
    def stop_solving(self):
        pg.quit()

    # Solving the problem with genetic
    @pyqtSlot()
    def solve_with_genetic(self):
        n = 0
        try:
            n = int(self.textbox.text())
            mr = 0
            pop = 0
            gen = 0
            pop, done1 = QtWidgets.QInputDialog.getInt(
                self, 'Population', 'Enter Population Size (300 is preferred):')
            mr, done2 = QtWidgets.QInputDialog.getDouble(
                self, 'Mutation', 'Enter Mutation Rate (0.1 is preferred):')

            gen, done3 = QtWidgets.QInputDialog.getDouble(
                self, 'Generation', 'Enter The Maximum Generation Number (100 is preferred):')

            if done1 and done2 and done3:
                mr = float(mr)
                pop = int(pop)
                gen = int(gen)

                stop = QPushButton('Close', self)
                stop.resize(stop.sizeHint())
                stop.clicked.connect(self.stop_solving)
                self.mainLayout.addWidget(stop)

                # start search
                gSolver = GeneticSolver(n, pop, mr, gen)
                steps, time = gSolver.solve()

                stepsW = QLabel("Genetic Steps: {}".format(steps), self)
                stepsW.setAlignment(QtCore.Qt.AlignCenter)

                timeW = QLabel("Genetic Time: {}".format(round(time)), self)
                timeW.setAlignment(QtCore.Qt.AlignCenter)

                self.mainLayout.addWidget(timeW)
                self.mainLayout.addWidget(stepsW)
        except:
            pass

    # Solving the problem with backtracking
    @pyqtSlot()
    def solve_with_backtracking(self):
        n = 0
        try:
            n = int(self.textbox.text())
            stop = QPushButton('Close', self)
            stop.resize(stop.sizeHint())
            stop.clicked.connect(self.stop_solving)
            self.mainLayout.addWidget(stop)

            # start search
            bSolver = BacktrackingSolver(n, mrv=self.mrv.isChecked(), mcv=self.mcv.isChecked(), lcv=self.lcv.isChecked(),
                                    fc=self.fc.isChecked(), arc=self.arc.isChecked())

            time = bSolver.solve()

            # stepsW = QLabel("Genetic Steps: {}".format(steps), self)
            # stepsW.setAlignment(QtCore.Qt.AlignCenter)

            timeW = QLabel("Genetic Time: {}".format(round(time)), self)
            timeW.setAlignment(QtCore.Qt.AlignCenter)

            self.mainLayout.addWidget(timeW)
            # self.mainLayout.addWidget(stepsW)

        except:
            pass


def main():
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()