import itertools
import pygame as pg
import random as rn
import numpy as np
import timeit
import copy

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