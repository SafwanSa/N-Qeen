import sys
from PyQt5.QtWidgets import (QWidget, QToolTip,
                             QPushButton, QApplication, QLabel, QVBoxLayout, QLineEdit, QCheckBox, QRadioButton,
                             QHBoxLayout)
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
import pygame as pg
from AStarSolver import AStarSolver
from GeneticSolver import GeneticSolver
from Backtracking import BacktrackingSolver

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