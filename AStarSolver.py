import itertools
import pygame as pg
import numpy as np
import timeit
from queue import PriorityQueue
import copy
import random as rn

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