from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import stats
from sklearn import datasets, linear_model
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_rb(object):
    def setupUi(self, Dialog_rb):
        Dialog_rb.setObjectName("Dialog_rb")
        Dialog_rb.resize(318, 206)
        self.radioButton = QtWidgets.QRadioButton(Dialog_rb)
        self.radioButton.setGeometry(QtCore.QRect(90, 60, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(Dialog_rb)
        self.radioButton_2.setGeometry(QtCore.QRect(90, 100, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(Dialog_rb)
        self.radioButton_3.setGeometry(QtCore.QRect(90, 150, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.label = QtWidgets.QLabel(Dialog_rb)
        self.label.setGeometry(QtCore.QRect(20, 0, 271, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(Dialog_rb)
        self.frame.setGeometry(QtCore.QRect(0, 0, 321, 211))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.raise_()
        self.radioButton.raise_()
        self.radioButton_2.raise_()
        self.radioButton_3.raise_()
        self.label.raise_()

        self.retranslateUi(Dialog_rb)
        QtCore.QMetaObject.connectSlotsByName(Dialog_rb)


    def retranslateUi(self, Dialog_rb):
        _translate = QtCore.QCoreApplication.translate
        Dialog_rb.setWindowTitle(_translate("Dialog_rb", "Регрессии"))
        self.radioButton.setText(_translate("Dialog_rb", "Линейная"))
        self.radioButton_2.setText(_translate("Dialog_rb", "Кубическая"))
        self.radioButton_3.setText(_translate("Dialog_rb", "Квадратическая"))
        self.label.setText(_translate("Dialog_rb", "Регрессии"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_rb = QtWidgets.QDialog()
    ui = Ui_Dialog_rb()
    ui.setupUi(Dialog_rb)
    Dialog_rb.show()
    sys.exit(app.exec_())
