import sys


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import uic, QtWidgets, Qt
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QTableView, qApp, QFileDialog, QAction
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from calendar_dialog_2 import Ui_Calendar_Dialog
from graph import Ui_dialog_graph_1
from rb import Ui_Dialog_rb
from tax_calc import Ui_TaxMainWindow


class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            exitAction = QAction(QIcon('exit.jpg'), 'Exit', self)
            exitAction.setShortcut('Ctrl+Q')
            exitAction.triggered.connect(qApp.quit)

            printAction = QAction(QIcon('print.png'), 'Print', self)
            printAction.setShortcut('Ctrl+P')
            printAction.triggered.connect(self.printDialog)

            saveAction = QAction(QIcon('save.ico'), 'Save', self)
            saveAction.setShortcut('Ctrl+S')
            saveAction.triggered.connect(self.file_save)

            openAction = QAction(QIcon('open.png'), 'Open', self)
            openAction.setShortcut('Ctrl+O')
            openAction.triggered.connect(self.openfile)

            calAction = QAction(QIcon('calendar-icon.png'), 'Календарь', self)
            calAction.setShortcut('Ctrl+C')
            calAction.triggered.connect(self.paint_cell)

            taxAction = QAction(QIcon('nalog.png'), 'Калькулятор', self)
            taxAction.triggered.connect(self.tax_action)

            rbAction = QAction(QIcon('rb.png'), 'Радио кнопки', self)
            rbAction.triggered.connect(self.radio_buttons)


            uic.loadUiType("F:/Python/lab7/ui/Vlad.ui", self)
            self.setWindowTitle('Работа с CSV')
            self.title = 'Python program'
            self.pushButton_4.clicked.connect(qApp.quit)
            self.action_4.triggered.connect(qApp.quit)
            self.pushButton.clicked.connect(self.openfile)
            self.pushButton.setShortcut('Ctrl+O')
            self.pushButton_2.clicked.connect(self.file_save)
            self.action.triggered.connect(self.openfile)
            self.action_5.triggered.connect(self.data_gaps_is_null)
            self.actionPad.triggered.connect(self.data_gaps_pad)
            self.actionFill.triggered.connect(self.data_gaps_fill)
            self.actionAkima.triggered.connect(self.data_gaps_akima)
            self.actionPolynomial_2.triggered.connect(self.data_gaps_polynomial)
            self.actionSpline_2.triggered.connect(self.data_gaps_spline)

            self.action_1.triggered.connect(self.data_graph)
            self.action_7.triggered.connect(self.data_graph_2)
            self.action_8.triggered.connect(self.data_graph_3)
            self.action_9.triggered.connect(self.data_graph_4)
            self.action_10.triggered.connect(self.data_graph_5)
            self.action_6.triggered.connect(self.data_graph_6)
            self.action_11.triggered.connect(self.data_graph_7)
            self.action_12.triggered.connect(self.data_graph_8)
            self.action_14.triggered.connect(self.data_graph_9)

            self.toolbar = self.addToolBar('ToolBar')
            self.toolbar.addAction(exitAction)
            self.toolbar.addAction(saveAction)
            self.toolBar.addAction(calAction)
            self.toolBar.addAction(taxAction)
            self.toolBar.addAction(rbAction)
            self.toolbar.addAction(openAction)

            self.show()

        def file_save(self):
            response = QFileDialog.getSaveFileName(
            parent=self,
            caption='Сохранение',
            directory='../SAVE/data.csv',
            filter='Data File(*.csv)',
            initialFilter='CSV File (*.csv)'
            )
            print(response)
            return response[0]

        def data_gaps_is_null(self):
            model = pandasModel(df.isnull())
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def data_gaps_pad(self):
            model = pandasModel(df.fillna(method='pad'))
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def data_gaps_fill(self):
            model = pandasModel(df.fillna(0))
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def data_gaps_akima(self):
            model = pandasModel(df.interpolate(method='akima').head(9))
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def data_gaps_polynomial(self):
            model = pandasModel(df.interpolate(method='polynomial', order=2))
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def data_gaps_spline(self):
            model = pandasModel(df.interpolate(method="spline", order=3))
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def openfile(self):
            fname = QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', "Image files (*.jpg *.csv)")
            global df
            f = open(fname[0], 'r')
            df = pd.read_csv(f)
            model = pandasModel(df)
            view = QTableView()
            view.setModel(model)
            self.tableView.setModel(model)

            model_new = pandasModel(df.describe())
            view = QTableView()
            view.setModel(model_new)
            self.tableView_3.setModel(model_new)

        def printDialog(self):
            printer = QPrinter(QPrinter.HighResolution)
            dialog = QPrintDialog(printer, self)
            if dialog.exec_() == QPrintDialog.Accepted:
                self.textEdit.print_(printer)

        def tax_action(self):
            global TaxMainWindow
            TaxMainWindow = QtWidgets.QMainWindow()
            ui = Ui_TaxMainWindow()
            ui.setupUi(TaxMainWindow)
            TaxMainWindow.show()

            def CalculateTax():
                price = int(ui.price_box.toPlainText())
                tax = (ui.tax_rate.value())
                total_price = price + ((tax / 100) * price)
                total_price_string = "Стоимость с учетом налога составит: " + str(total_price)
                ui.results_window.setText(total_price_string)

            ui.calc_tax_button.clicked.connect(CalculateTax)

        def paint_cell(self):
            global Calendar_Dialog
            Calendar_Dialog = QtWidgets.QDialog()
            ui = Ui_Calendar_Dialog()
            ui.setupUi(Calendar_Dialog)
            Calendar_Dialog.show()

        def radio_buttons(self):
            global Dialog_rb
            Dialog_rb = QtWidgets.QDialog()
            ui = Ui_Dialog_rb()
            ui.setupUi(Dialog_rb)
            Dialog_rb.show()

            def Regr1():
                df = pd.read_csv("data13.csv")

                X = df[['Runtime']].values
                y = df['IMDB Score'].values

                regr = LinearRegression()

                # fit features
                X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

                regr = regr.fit(X, y)
                y_lin_fit = regr.predict(X_fit)
                linear_r2 = r2_score(y, regr.predict(X))

                # plot results
                plt.scatter(X, y, label='Данные', color='lightgray')

                plt.plot(X_fit, y_lin_fit,
                         label='Линейная (d=1), $R^2={:.2f}$'.format(linear_r2),
                         color='blue',
                         lw=2,
                         linestyle=':')

                plt.title('Линейная регрессия')
                plt.xlabel('Inputs')
                plt.ylabel('Predicted Values')
                plt.legend(loc='lower right')
                plt.show()

                d = {'Уравнение': ['Линейное (d=1)'], 'Коэф. дет.': ['R^2={:.2f}'.format(linear_r2)]}
                df = pd.DataFrame(data=d)
                model = pandasModel(df)
                view = QTableView()
                view.setModel(model)
                self.tableView_2.setModel(model)

            def Regr2():
                df = pd.read_csv("data13.csv")

                X = df[['Inputs']].values
                y = df['Height'].values

                regr = LinearRegression()

                # create quadratic features
                quadratic = PolynomialFeatures(degree=2)
                cubic = PolynomialFeatures(degree=3)
                X_quad = quadratic.fit_transform(X)
                X_cubic = cubic.fit_transform(X)

                # fit features
                X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

                regr = regr.fit(X_cubic, y)
                y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
                cubic_r2 = r2_score(y, regr.predict(X_cubic))

                # plot results
                plt.scatter(X, y, label='Данные', color='lightgray')

                plt.plot(X_fit, y_cubic_fit,
                         label='Кубическая (d=3), $R^2={:.2f}$'.format(cubic_r2),
                         color='green',
                         lw=2,
                         linestyle='--')

                plt.title('Кубическая регрессия')
                plt.xlabel('Уголь')
                plt.ylabel('Масло')
                plt.legend(loc='lower right')
                plt.show()

                d = {'Уравнение': ['Кубическое (d=2)'], 'Коэф. дет.': ['R^2={:.2f}'.format(cubic_r2)]}
                df = pd.DataFrame(data=d)
                model = pandasModel(df)
                view = QTableView()
                view.setModel(model)
                self.tableView_2.setModel(model)

            def Regr3():
                df = pd.read_csv("data13_2.csv.csv")

                X = df[['price']].values
                y = df['max_price'].values

                regr = LinearRegression()

                # create quadratic features
                quadratic = PolynomialFeatures(degree=2)
                cubic = PolynomialFeatures(degree=3)
                X_quad = quadratic.fit_transform(X)
                X_cubic = cubic.fit_transform(X)

                # fit features
                X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

                regr = regr.fit(X_quad, y)
                y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
                quadratic_r2 = r2_score(y, regr.predict(X_quad))

                # plot results
                plt.scatter(X, y, label='Данные', color='lightgray')

                plt.plot(X_fit, y_quad_fit,
                         label='Квадратичная (d=2), $R^2={:.2f}$'.format(quadratic_r2),
                         color='red',
                         lw=2,
                         linestyle='-')

                plt.title('Квадратичная регрессия')
                plt.xlabel('Уголь')
                plt.ylabel('Масло')
                plt.legend(loc='lower right')
                plt.show()

                d = {'Уравнение': ['Квадратическое (d=3)'], 'Коэф. дет.': ['R^2={:.2f}'.format(quadratic_r2)]}
                df = pd.DataFrame(data=d)
                model = pandasModel(df)
                view = QTableView()
                view.setModel(model)
                self.tableView_2.setModel(model)

            ui.radioButton.clicked.connect(Regr1)
            ui.radioButton_2.clicked.connect(Regr2)
            ui.radioButton_3.clicked.connect(Regr3)

        def data_graph(self):
            df = pd.read_csv('data13_2.csv')
            sns.set_style('whitegrid')
            sns.countplot(x='max_price', data=df)
            plt.show()

        def data_graph_2(self):
            df = pd.read_csv('data13_2.csv')
            sns.displot(df['position'], kde=False)
            plt.show()

        def data_graph_3(self):
            df = pd.read_csv('data13_2.csv')[2:17]
            x = df['price']
            y = df['max_price']
            plt.title('price to max_price')
            plt.xlabel('price')
            plt.ylabel('max_price')
            plt.bar(x, y, width=4)
            plt.show()

        def data_graph_4(self):
            df = pd.read_csv('data13_2.csv')
            x_age = df['age']
            y_bwt = df['height']
            plt.title('age and height')
            plt.xlabel('age')
            plt.ylabel('height')
            plt.scatter(x_age, y_bwt, )
            plt.show()

        def data_graph_5(self):
            df = pd.read_csv('data13_2.csv')
            sns.countplot(x='price', hue='max_price', data = df)
            plt.show()

        def data_graph_6(self):
            data = pd.read_csv("data13_2.csv")
            data2 = data.sort_values(by='max_price').copy()
            x = data2.iloc[:, 2:3]
            y = data2.iloc[:, 2:3]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=38)
            poly_reg = PolynomialFeatures(degree=5)
            x_poly = poly_reg.fit_transform(x)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(x_poly, y)
            plt.style.use('seaborn')
            plt.scatter(x, y, color="red", marker='o', s=35, alpha=0.5, label='Test data')
            plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color="blue", label='Model Plot')
            plt.title('Predicted Values vs Inputs')
            plt.xlabel('Inputs')
            plt.ylabel('Predicted Values')
            plt.legend(loc='upper left')
            plt.show()

        def graph_regress(self):
            data = pd.read_csv("data13_2.csv")
            data2 = data.sort_values(by='max_price').copy()
            x = data2.iloc[:, 2:3]
            y = data2.iloc[:, 2:3]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=38)
            poly_reg = PolynomialFeatures(degree=5)
            x_poly = poly_reg.fit_transform(x)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(x_poly, y)
            plt.style.use('seaborn')
            plt.scatter(x, y, color="red", marker='o', s=35, alpha=0.5, label='Test data')
            plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color="blue", label='Model Plot')
            plt.title('Predicted Values vs Inputs')
            plt.xlabel('Inputs')
            plt.ylabel('Predicted Values')
            plt.legend(loc='upper left')
            plt.show()

            X = data['price']
            y = data['max_price']
            y = data.iloc[:, -1]
            X = np.array(data['price']).reshape(-1, 1)
            y = np.array(data['max_price']).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            model = LinearRegression()
            model.fit(X_train, y_train)
            model = LinearRegression().fit(X_train, y_train)
            r_sq = model.score(X_train, y_train)
            y_pred = model.predict(y_test)

            # d = {'Уравнение': ['Линейное (d=1)'],'Коэф. дет.': ['R^2={:.2f}'.format(linear_r2)]}
            d = {'Уравнение': ['Линейное (d=1)', r_sq, model.intercept_, model.coef_], 'Коэф. дет.': ['R^2={:.2f}'.format(linear_r2), '-', '-', '-']}
            auto_types = (y_test)
            auto_df = pd.DataFrame(auto_types, columns=[['Пред. знач.']])
            df = pd.DataFrame(data=d)
            df1 = pd.DataFrame(data=auto_df)
            model = pandasModel(df)
            model1 = pandasModel(df1)
            view = QTableView()
            view.setModel(model)
            view = QTableView()
            view.setModel(model1)
            self.tableView_2.setModel(model)
            self.tableView.setModel(model1)

        def data_graph_7(self):
            df = pd.read_csv("data13_2.csv")

            X = df[['price']].values
            y = df['max_price'].values

            regr = LinearRegression()

            # create quadratic features
            quadratic = PolynomialFeatures(degree=2)
            cubic = PolynomialFeatures(degree=3)
            X_quad = quadratic.fit_transform(X)
            X_cubic = cubic.fit_transform(X)

            # fit features
            X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

            regr = regr.fit(X_cubic, y)
            y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
            cubic_r2 = r2_score(y, regr.predict(X_cubic))

            # plot results
            plt.scatter(X, y, label='Данные', color='lightgray')

            plt.plot(X_fit, y_cubic_fit,
                     label='Кубеческая (d=3), $R^2={:.2f}$'.format(cubic_r2),
                     color='green',
                     lw=2,
                     linestyle='--')

            plt.title('Кубическая регрессия')
            plt.xlabel('Уголь')
            plt.ylabel('Масло')
            plt.legend(loc='lower right')
            plt.show()

            d = {'Уравнение': ['Кубическое (d=3)'], 'Коэф. дет.': ['R^2={:.2f}'.format(cubic_r2)]}
            df = pd.DataFrame(data=d)
            model = pandasModel(df)
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def data_graph_8(self):
            df = pd.read_csv("data13_2.csv")

            X = df[['price']].values
            y = df['max_price'].values

            regr = LinearRegression()

            # create quadratic features
            quadratic = PolynomialFeatures(degree=2)
            cubic = PolynomialFeatures(degree=3)
            X_quad = quadratic.fit_transform(X)
            X_cubic = cubic.fit_transform(X)

            # fit features
            X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

            regr = regr.fit(X_quad, y)
            y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
            quadratic_r2 = r2_score(y, regr.predict(X_quad))

            # plot results
            plt.scatter(X, y, label='Данные', color='lightgray')

            plt.plot(X_fit, y_quad_fit,
                     label='Квадратичная (d=2), $R^2={:.2f}$'.format(quadratic_r2),
                     color='red',
                     lw=2,
                     linestyle='-')

            plt.title('Квадратичная регрессия')
            plt.xlabel('Уголь')
            plt.ylabel('Масло')
            plt.legend(loc='lower right')
            plt.show()

            d = {'Уравнение': ['Квадратическое (d=2)'], 'Коэф. дет.': ['R^2={:.2f}'.format(quadratic_r2)]}
            df = pd.DataFrame(data=d)
            model = pandasModel(df)
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        #Общий график
        def data_graph_9(self):
            df = pd.read_csv("data13_2.csv")
            X = df[['price']].values
            y = df['max_price'].values

            regr = LinearRegression()

            # create quadratic features
            quadratic = PolynomialFeatures(degree=2)
            cubic = PolynomialFeatures(degree=3)
            X_quad = quadratic.fit_transform(X)
            X_cubic = cubic.fit_transform(X)

            # fit features
            X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

            regr = regr.fit(X, y)
            y_lin_fit = regr.predict(X_fit)
            linear_r2 = r2_score(y, regr.predict(X))

            regr = regr.fit(X_quad, y)
            y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
            quadratic_r2 = r2_score(y, regr.predict(X_quad))

            regr = regr.fit(X_cubic, y)
            y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
            cubic_r2 = r2_score(y, regr.predict(X_cubic))

            # plot results
            plt.scatter(X, y, label='Данные', color='lightgray')

            plt.plot(X_fit, y_lin_fit,
                     label='Линейная (d=1), $R^2={:.2f}$'.format(linear_r2),
                     color='blue',
                     lw=2,
                     linestyle=':')

            plt.plot(X_fit, y_quad_fit,
                     label='Квадратичная (d=2), $R^2={:.2f}$'.format(quadratic_r2),
                     color='red',
                     lw=2,
                     linestyle='-')

            plt.plot(X_fit, y_cubic_fit,
                     label='Кубическая (d=3), $R^2={:.2f}$'.format(cubic_r2),
                     color='green',
                     lw=2,
                     linestyle='--')

            plt.xlabel('Уголь')
            plt.title('Регрессия общий график')
            plt.ylabel('Масло')
            plt.legend(loc='lower right')
            plt.show()

            d = {'Уравнение': ['Квадратическое (d=3)', 'Кубическое (d=3)', 'Линейное (d=1)'], 'Коэф. дет.': ['R^2={:.2f}'.format(quadratic_r2), 'R^2={:.2f}'.format(cubic_r2), 'R^2={:.2f}'.format(linear_r2)]}
            df = pd.DataFrame(data=d)
            model = pandasModel(df)
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Win = Window()
    app.exec_()
