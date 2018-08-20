# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'jpeg.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QProgressBar, QWidget
from PyQt5.QtGui import QPixmap, QImage
from Compression import JPEG
import qimage2ndarray
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import PIL as pil
from PIL import Image, ImageQt

JPEG_Kompression, RGB_Bild, Y_Bild, Cb_Bild, Cr_Bild, Y_DCT_Koeffizienten, Cb_DCT_Koeffizienten, Cr_DCT_Koeffizienten, Y_Quant_DCT_Koeffizienten, Cb_Quant_DCT_Koeffizienten, Cr_Quant_DCT_Koeffizienten, JPEG_RGB_Bild, Vergleich = "JPEG Kompression", "RGB Bild", "Y Bild", "Cb Bild", "Cr Bild", "Y DCT Koeffizienten", "Cb DCT Koeffizienten", "Cr DCT Koeffizienten", "Y Quant. DCT Koeffizienten", "Cb Quant. DCT Koeffizienten", "Cr Quant. DCT Koeffizienten", "JPEG RGB Bild", "Vergleich"

Y_Dequant_DCT_Koeffizienten, Cb_Dequant_DCT_Koeffizienten, Cr_Dequant_DCT_Koeffizienten, Y_IDCT_Koeffizienten, Cb_IDCT_Koeffizienten, Cr_IDCT_Koeffizienten = "Y Dequant. DCT Koeffizienten", "Cb Dequant. DCT Koeffizienten", "Cr Dequant. DCT Koeffizienten", "Y IDCT Koeffizienten", "Cb IDCT Koeffizienten", "Cr IDCT Koeffizienten"

werte = [
    RGB_Bild, Y_Bild, Cb_Bild, Cr_Bild, Y_DCT_Koeffizienten,
    Cb_DCT_Koeffizienten, Cr_DCT_Koeffizienten, Y_Quant_DCT_Koeffizienten,
    Cb_Quant_DCT_Koeffizienten, Cr_Quant_DCT_Koeffizienten,
    Y_Dequant_DCT_Koeffizienten, Cb_Dequant_DCT_Koeffizienten,
    Cr_Dequant_DCT_Koeffizienten, Y_IDCT_Koeffizienten, Cb_IDCT_Koeffizienten,
    Cr_IDCT_Koeffizienten, JPEG_RGB_Bild, Vergleich
]


class Ui_MainWindow(object):
    def new(self):

        fileName, _ = QFileDialog.getOpenFileName(
            self.menuNew, "", "../Testbilder", "", "PNG (*.png)")
        if not fileName:
            return ''

        w = ProgressPopup()
        # w.setGeometry(QtCore.QRect(100, 100, 400, 200))
        w.show()

        self.nvm = JPEG(fileName)
        self.histmaps = []
        self.bildmaps = []

        for i in range(0, len(self.nvm.werte)):
            self.histmaps.append(self.nvm.histogramm(self.nvm.werte[i]))
            self.bildmaps.append(self.nvm.drucke_bild(str(i), self.nvm.werte[i]))


        self.centralwidget.setEnabled(True)

    def clicked(self, item):
        i = False
        index = werte.index(item.text())

        matrix = self.nvm.werte[index]

        qll = self.nvm.quellenredundanz(matrix)
        entsg = self.nvm.entscheidungsgehalt(matrix)
        entr = self.nvm.entropie(matrix)
       
        # hist = self.nvm.drucke_tmp_histogramm(matrix)
        # bild = self.nvm.drucke_tmp_bild(matrix)

        self.label_entropie.setText(str(entr))
        self.label_quellenredundanz.setText(str(qll))
        # self.label_entscheidungsgehalt.setText(entsg)

        self.label_left.setPixmap(QPixmap(self.bildmaps[index]))
        self.label_right.figure = self.histmaps[index]

        self.label_right.draw()
        i = True
        if index == len(werte) - 1:
            psnr = self.nvm.PSNR()
            compfak = self.nvm.compFak()
            size1, size2 = self.nvm.groessen()

            self.label_psnr.setText(str(psnr))
            self.label_groesse.setText(str(compfak))
            

    # def testColorMap(self, cmap):
        # sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
        # fig = Figure((2.5,.2), subplotpars = sp)
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # gradient = np.linspace(0, 1, 256)
        # gradient = np.vstack((gradient, gradient))
        # ax.set_axis_off()
        # canvas.draw()
        # size = canvas.size()
        # width, height = size.width(), size.height()
        # im = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        # return QPixmap(im)


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1019, 614)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("resources/jpeg.png"), QtGui.QIcon.Normal,
            QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setLocale(
            QtCore.QLocale(QtCore.QLocale.German, QtCore.QLocale.Germany))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(False)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setObjectName("listWidget")
        self.listWidget.currentItemChanged.connect(self.clicked)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.horizontalLayout.addWidget(self.listWidget)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_left = QtWidgets.QLabel(self.tab)
        self.label_left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_left.setText("")
        self.label_left.setScaledContents(False)
        self.label_left.setObjectName("label_left")
        self.horizontalLayout_2.addWidget(self.label_left)
        spacerItem = QtWidgets.QSpacerItem(20, 198,
                                           QtWidgets.QSizePolicy.Minimum,
                                           QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem)

        self.figure = Figure()
        self.label_right = FigureCanvas(self.figure)
        self.label_right.setObjectName("label_right")
        self.horizontalLayout_2.addWidget(self.label_right)
        self.horizontalLayout_2.setStretch(0, 50)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 50)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(618, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole,
                                    self.label_3)
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole,
                                    self.label_4)
        self.label_5 = QtWidgets.QLabel(self.tab)
        # self.label_5.setEnabled(False)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole,
                                    self.label_5)
        self.label_quellenredundanz = QtWidgets.QLabel(self.tab)
        self.label_quellenredundanz.setText("")
        self.label_quellenredundanz.setObjectName("label_quellenredundanz")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole,
                                    self.label_quellenredundanz)
        self.label_entropie = QtWidgets.QLabel(self.tab)
        self.label_entropie.setText("")
        self.label_entropie.setObjectName("label_entropie")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole,
                                    self.label_entropie)
        self.label_psnr = QtWidgets.QLabel(self.tab)
        self.label_psnr.setText("")
        self.label_psnr.setObjectName("label_psnr")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole,
                                    self.label_psnr)
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole,
                                    self.label_9)
        self.label_groesse = QtWidgets.QLabel(self.tab)
        self.label_groesse.setText("")
        self.label_groesse.setObjectName("label_groesse")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole,
                                    self.label_groesse)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.listWidget.raise_()
        self.label_left.raise_()
        self.label_right.raise_()
        self.listWidget.raise_()
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_2)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout_3.addWidget(self.textBrowser)
        self.tabWidget.addTab(self.tab_2, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.horizontalLayout.setStretch(0, 10)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 50)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1019, 21))
        self.menubar.setObjectName("menubar")
        self.menuNew = QtWidgets.QMenu(self.menubar)
        self.menuNew.setObjectName("menuNew")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNeu = QtWidgets.QAction(MainWindow)
        self.actionNeu.setObjectName("actionNeu")
        self.actionSchliessen = QtWidgets.QAction(MainWindow)
        self.actionSchliessen.setObjectName("actionSchliessen")
        self.actionQuelle = QtWidgets.QAction(MainWindow)
        self.actionQuelle.setEnabled(False)
        self.actionQuelle.setObjectName("actionQuelle")
        self.menuNew.addAction(self.actionNeu)
        self.actionNeu.triggered.connect(self.new)
        self.menuNew.addAction(self.actionQuelle)
        self.menuNew.addSeparator()
        self.menuNew.addAction(self.actionSchliessen)
        self.menubar.addAction(self.menuNew.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", JPEG_Kompression))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", RGB_Bild))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", Y_Bild))
        item = self.listWidget.item(2)
        item.setText(_translate("MainWindow", Cb_Bild))
        item = self.listWidget.item(3)
        item.setText(_translate("MainWindow", Cr_Bild))
        item = self.listWidget.item(4)
        item.setText(_translate("MainWindow", Y_DCT_Koeffizienten))
        item = self.listWidget.item(5)
        item.setText(_translate("MainWindow", Cb_DCT_Koeffizienten))
        item = self.listWidget.item(6)
        item.setText(_translate("MainWindow", Cr_DCT_Koeffizienten))
        item = self.listWidget.item(7)
        item.setText(_translate("MainWindow", Y_Quant_DCT_Koeffizienten))
        item = self.listWidget.item(8)
        item.setText(_translate("MainWindow", Cb_Quant_DCT_Koeffizienten))
        item = self.listWidget.item(9)
        item.setText(_translate("MainWindow", Cr_Quant_DCT_Koeffizienten))
        item = self.listWidget.item(10)
        item.setText(_translate("MainWindow", Y_Dequant_DCT_Koeffizienten))
        item = self.listWidget.item(11)
        item.setText(_translate("MainWindow", Cb_Dequant_DCT_Koeffizienten))
        item = self.listWidget.item(12)
        item.setText(_translate("MainWindow", Cr_Dequant_DCT_Koeffizienten))
        item = self.listWidget.item(13)
        item.setText(_translate("MainWindow", Y_IDCT_Koeffizienten))
        item = self.listWidget.item(14)
        item.setText(_translate("MainWindow", Cb_IDCT_Koeffizienten))
        item = self.listWidget.item(15)
        item.setText(_translate("MainWindow", Cr_IDCT_Koeffizienten))
        item = self.listWidget.item(16)
        item.setText(_translate("MainWindow", Y_Bild))
        item = self.listWidget.item(17)
        item.setText(_translate("MainWindow", Cb_Bild))
        item = self.listWidget.item(18)
        item.setText(_translate("MainWindow", Cr_Bild))
        item = self.listWidget.item(19)
        item.setText(_translate("MainWindow", JPEG_RGB_Bild))
        item = self.listWidget.item(20)
        item.setText(_translate("MainWindow", Vergleich))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.label_3.setText(_translate("MainWindow", "Quellenredundanz"))
        self.label_4.setText(_translate("MainWindow", "Entropie"))
        self.label_5.setText(_translate("MainWindow", "PSNR"))
        self.label_9.setText(_translate("MainWindow", "Größe"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab),
            _translate("MainWindow", "Ergebnisse"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_2), _translate(
                "MainWindow", "Code"))
        self.menuNew.setTitle(_translate("MainWindow", "Menü"))
        self.actionNeu.setText(_translate("MainWindow", "Neue Kompression..."))
        self.actionNeu.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionSchliessen.setText(_translate("MainWindow", "Schließen"))
        self.actionSchliessen.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionQuelle.setText(_translate("MainWindow", "Quelle Anzeigen"))
        self.actionQuelle.setShortcut(_translate("MainWindow", "Ctrl+O"))


class ProgressPopup(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        QWidget.windowModality == QtCore.Qt.ApplicationModal
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 0)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
