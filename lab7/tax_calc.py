# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tax_calc.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TaxMainWindow(object):
    def setupUi(self, TaxMainWindow):
        TaxMainWindow.setObjectName("TaxMainWindow")
        TaxMainWindow.resize(600, 300)
        TaxMainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(TaxMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.price_box = QtWidgets.QTextEdit(self.centralwidget)
        self.price_box.setGeometry(QtCore.QRect(20, 40, 111, 41))
        self.price_box.setObjectName("price_box")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.tax_rate = QtWidgets.QSpinBox(self.centralwidget)
        self.tax_rate.setGeometry(QtCore.QRect(190, 40, 101, 41))
        self.tax_rate.setProperty("value", 20)
        self.tax_rate.setObjectName("tax_rate")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 10, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.results_window = QtWidgets.QTextEdit(self.centralwidget)
        self.results_window.setGeometry(QtCore.QRect(250, 190, 201, 101))
        self.results_window.setObjectName("results_window")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(130, 220, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.calc_tax_button = QtWidgets.QPushButton(self.centralwidget)
        self.calc_tax_button.setGeometry(QtCore.QRect(330, 40, 151, 41))
        self.calc_tax_button.setObjectName("calc_tax_button")
        TaxMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TaxMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 698, 21))
        self.menubar.setObjectName("menubar")
        TaxMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TaxMainWindow)
        self.statusbar.setObjectName("statusbar")
        TaxMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(TaxMainWindow)
        QtCore.QMetaObject.connectSlotsByName(TaxMainWindow)

    def retranslateUi(self, TaxMainWindow):
        _translate = QtCore.QCoreApplication.translate
        TaxMainWindow.setWindowTitle(_translate("TaxMainWindow", "Калькулятор"))
        self.label.setText(_translate("TaxMainWindow", "Стоимость"))
        self.label_2.setText(_translate("TaxMainWindow", "% налога"))
        self.label_4.setText(_translate("TaxMainWindow", "Результат"))
        self.calc_tax_button.setText(_translate("TaxMainWindow", "Вычислить"))