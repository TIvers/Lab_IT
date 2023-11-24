import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog_graph_1(object):

    def setupUi(self, dialog_graph_1):
        dialog_graph_1.setObjectName("1")
        dialog_graph_1.resize(1, 1)

        data = pd.read_csv('4_2.csv')

        fig, ax = plt.subplots()

        x = data['age']
        y = data['height']

        ax.plot(x, y, label='Объем')
        ax.legend()
        ax.set_title('age to height')
        plt.grid(True)
        plt.show()