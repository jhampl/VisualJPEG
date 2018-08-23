# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\popup.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ProgressPopup(object):
    def setupUi(self, ProgressPopup):
        ProgressPopup.setObjectName("ProgressPopup")
        ProgressPopup.setWindowModality(QtCore.Qt.ApplicationModal)
        ProgressPopup.resize(204, 48)
        ProgressPopup.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        ProgressPopup.setWindowTitle("Lade...")
        self.horizontalLayout = QtWidgets.QHBoxLayout(ProgressPopup)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.progressBar = QtWidgets.QProgressBar(ProgressPopup)
        self.progressBar.setMaximum(0)
        self.progressBar.setProperty("value", -1)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout.addWidget(self.progressBar)

        self.retranslateUi(ProgressPopup)
        QtCore.QMetaObject.connectSlotsByName(ProgressPopup)

    def retranslateUi(self, ProgressPopup):
        pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ProgressPopup = QtWidgets.QWidget()
    ui = Ui_ProgressPopup()
    ui.setupUi(ProgressPopup)
    ProgressPopup.show()
    sys.exit(app.exec_())

