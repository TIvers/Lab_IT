# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calendar_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Calendar_Dialog(object):
    def setupUi(self, Calendar_Dialog):
        Calendar_Dialog.setObjectName("Calendar_Dialog")
        Calendar_Dialog.resize(400, 300)
        self.calendarWidget = QtWidgets.QCalendarWidget(Calendar_Dialog)
        self.calendarWidget.setGeometry(QtCore.QRect(0, 0, 401, 301))
        self.calendarWidget.setObjectName("calendarWidget")

        self.retranslateUi(Calendar_Dialog)
        QtCore.QMetaObject.connectSlotsByName(Calendar_Dialog)

    def retranslateUi(self, Calendar_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Calendar_Dialog.setWindowTitle(_translate("Calendar_Dialog", "Календарь"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Calendar_Dialog = QtWidgets.QDialog()
    ui = Ui_Calendar_Dialog()
    ui.setupUi(Calendar_Dialog)
    Calendar_Dialog.show()
    sys.exit(app.exec_())
