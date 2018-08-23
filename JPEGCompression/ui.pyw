# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'jpeg.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
import Compression as nvm
from popup import Ui_ProgressPopup
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

RGB_Bild, R_Bild, G_Bild, B_Bild = "PNG Bild", "PNG-R Bild", "PNG-G Bild", "PNG-B Bild"
YCbCr_Bild, Y_Bild, Cb_Bild, Cr_Bild = "YCbCr Bild", "Y Bild", "Cb Bild", "Cr Bild"
YCbCr_sub_Bild, Y_sub_Bild, Cb_sub_Bild, Cr_sub_Bild = "Downsampling YCbCr Bild", "Downsampling Y Bild", "Downsampling Cb Bild", "Downsampling Cr Bild"
Y_DCT_Koeffizienten, Cb_DCT_Koeffizienten, Cr_DCT_Koeffizienten = "Y DCT Koeffizienten", "Cb DCT Koeffizienten", "Cr DCT Koeffizienten"
Y_Quant_DCT_Koeffizienten, Cb_Quant_DCT_Koeffizienten, Cr_Quant_DCT_Koeffizienten = "Quant. Y DCT Koeffizienten", "Quant. Cb DCT Koeffizienten", "Quant. Cr DCT Koeffizienten"
Y_Cod_DCT_Koeffizienten, Cb_Cod_DCT_Koeffizienten, Cr_Cod_DCT_Koeffizienten = "Codierte Y DCT Koeffizienten", "Codierte Cb DCT Koeffizienten", "Codierte Cr DCT Koeffizienten"
Y_Decod_DCT_Koeffizienten, Cb_Decod_DCT_Koeffizienten, Cr_Decod_DCT_Koeffizienten = "Decodierte Y DCT Koeffizienten", "Decodierte Cb DCT Koeffizienten", "Decodierte Cr DCT Koeffizienten"
Y_Dequant_DCT_Koeffizienten, Cb_Dequant_DCT_Koeffizienten, Cr_Dequant_DCT_Koeffizienten = "Dequant. Y DCT Koeffizienten", "Dequant. Cb DCT Koeffizienten", "Dequant. Cr DCT Koeffizienten"
Y_IDCT_Koeffizienten, Cb_IDCT_Koeffizienten, Cr_IDCT_Koeffizienten = "Y IDCT Koeffizienten", "Cb IDCT Koeffizienten", "Cr IDCT Koeffizienten"
YCbCr_up_Bild, Y_up_Bild, Cb_up_Bild, Cr_up_Bild = "Upsampling YCbCr Bild", "Upsampling Y Bild", "Upsampling Cb Bild", "Upsampling Cr Bild"
JPEG_RGB_Bild, JPEG_R_Bild, JPEG_G_Bild, JPEG_B_Bild = "JPEG Bild", "JPEG-R Bild", "JPEG-G Bild", "JPEG-B Bild"
Vergleich_Quant, Vergleich_Bilder = "Vergleich mit Quantisierung", "Vergleich mit rekonstr. Bild"

labels = [
    RGB_Bild, R_Bild, G_Bild, B_Bild, YCbCr_Bild, Y_Bild, Cb_Bild, Cr_Bild,
    Y_sub_Bild, Cb_sub_Bild, Cr_sub_Bild,
    Y_DCT_Koeffizienten, Cb_DCT_Koeffizienten, Cr_DCT_Koeffizienten,
    Y_Quant_DCT_Koeffizienten, Cb_Quant_DCT_Koeffizienten, Cr_Quant_DCT_Koeffizienten,
    Y_Dequant_DCT_Koeffizienten, Cb_Dequant_DCT_Koeffizienten, Cr_Dequant_DCT_Koeffizienten,
    Y_Cod_DCT_Koeffizienten, Cb_Cod_DCT_Koeffizienten, Cr_Cod_DCT_Koeffizienten,
    Y_Decod_DCT_Koeffizienten, Cb_Decod_DCT_Koeffizienten, Cr_Decod_DCT_Koeffizienten,
    Y_IDCT_Koeffizienten, Cb_IDCT_Koeffizienten, Cr_IDCT_Koeffizienten,
    Y_up_Bild, Cb_up_Bild, Cr_up_Bild,
    JPEG_RGB_Bild, JPEG_R_Bild, JPEG_G_Bild, JPEG_B_Bild,
    Vergleich_Quant, Vergleich_Bilder
]

wertemap = []
switch = True
histmaps = []
bildmaps = []
qll, entsg, entr, index, psnr, compfak, groesse1, groesse2, mse = 0, 0, 0, 0, 0, 0, 0, 0, 0


class Ui_MainWindow(object):
    def new(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.menuNew, "", "../Testbilder", "", "PNG (*.png)")
        if not fileName:
            return ''

        self.thread = JPEGThread(fileName)
        self.thread.signal.connect(self.done)
        self.thread.start()

        self.popup = QtWidgets.QWidget()
        ui = Ui_ProgressPopup()
        ui.setupUi(self.popup)
        self.popup.show()

    def clicked(self, item):
        global index
        index = labels.index(item.text())

        self.thread = LoadThread(index, self.popup)
        self.thread.signal.connect(self.setui)
        self.thread.start()

        self.popup = QtWidgets.QWidget()
        ui = Ui_ProgressPopup()
        ui.setupUi(self.popup)
        self.popup.show()

    def done(self):
        self.popup.close()
        self.centralwidget.setEnabled(True)

    def setui(self):

        if index < len(wertemap) - 2:
            self.stack.setCurrentIndex(0)
            self.label_3.setText("Quellenredundanz (bit/Symbol)")
            self.label_4.setText("Entropie (bit/Symbol)")
            self.label_5.setText("Entscheidungsgehalt")
            self.label_10.setText("Groesse (bytes)")
            self.label_9.setText("")

            self.label_compfak.setText("")
            self.label_entropie.setText(str(entr))
            self.label_quellenredundanz.setText(str(qll))
            self.label_psnr.setText(str(entsg))
            self.label_mse.setText(str(groesse1))

            self.label_left.setPixmap(QPixmap(nvm.drucke_bild(str(index), wertemap[index])))
            self.canvas.figure = nvm.histogramm(wertemap[index][0])

            self.canvas.draw()

        else:
            if index < len(wertemap) - 1:
                self.label_3.setText("Groesse Original, links (bytes)")
                self.label_4.setText("Groesse JPG, rechts (bytes)")
                self.label_5.setText("PSNR (dB)")
                self.label_10.setText("MSE (dB)")
                self.label_mse.setText(str(mse))
                self.label_psnr.setText(str(psnr))

            else:
                self.label_3.setText("Groesse Original, links (bytes)")
                self.label_4.setText("Groesse Codierte DCT Koeff., rechts (bytes)")
                self.label_5.setText("")
                self.label_10.setText("")

            self.stack.setCurrentIndex(1)
            self.label_9.setText("Kompressionsfaktor")
            self.label_quellenredundanz.setText(str(groesse1))
            self.label_entropie.setText(str(groesse2))
            self.label_compfak.setText(str(compfak))

            self.label_left.setPixmap(QPixmap(nvm.drucke_bild(str(index), wertemap[index][0])))
            self.label_right.setPixmap(QPixmap(nvm.drucke_bild(str(index), wertemap[index][1])))

        self.popup.close()

    def setupUi(self, MainWindow):
        self.window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1419, 914)
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

        for i in range(len(labels)):
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
        self.label_left.setText("")
        self.label_left.setScaledContents(False)
        self.label_left.setObjectName("label_left")
        self.horizontalLayout_2.addWidget(self.label_left)
        spacerItem = QtWidgets.QSpacerItem(20, 198,
                                           QtWidgets.QSizePolicy.Minimum,
                                           QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem)

        self.figure = Figure()

        self.stack = QtWidgets.QStackedWidget(self.tab)
        self.stack.setObjectName("stack")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.horizontalLayout_5 = QtWidgets.QVBoxLayout(self.page)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("canvas")
        self.horizontalLayout_5.addWidget(self.canvas)
        self.stack.addWidget(self.page)
        self.page2 = QtWidgets.QWidget()
        self.page2.setObjectName("page2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_right = QtWidgets.QLabel(self.page)
        self.label_right.setObjectName("label_right")
        self.verticalLayout_2.addWidget(self.label_right)
        self.stack.addWidget(self.page2)
        self.horizontalLayout_2.addWidget(self.stack)

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
        self.label_10 = QtWidgets.QLabel(self.tab)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole,
                                    self.label_10)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole,
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
        self.label_mse = QtWidgets.QLabel(self.tab)
        self.label_mse.setText("")
        self.label_mse.setObjectName("label_mse")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole,
                                    self.label_mse)
        self.label_psnr = QtWidgets.QLabel(self.tab)
        self.label_psnr.setText("")
        self.label_psnr.setObjectName("label_psnr")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole,
                                    self.label_psnr)
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole,
                                    self.label_9)
        self.label_compfak = QtWidgets.QLabel(self.tab)
        self.label_compfak.setText("")
        self.label_compfak.setObjectName("label_compfak")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole,
                                    self.label_compfak)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.horizontalLayout.addWidget(self.tabWidget)
        self.tabWidget.addTab(self.tab, "")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_2)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout_3.addWidget(self.textBrowser)
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
        MainWindow.setWindowTitle(_translate("MainWindow", "JPEG Kompression"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)

        for i in range(len(labels)):
            item = self.listWidget.item(i)
            item.setText(_translate("MainWindow", labels[i]))

        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.label_3.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", ""))
        self.label_5.setText(_translate("MainWindow", ""))
        self.label_9.setText(_translate("MainWindow", ""))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Ergebnisse"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Code"))
        self.menuNew.setTitle(_translate("MainWindow", "Menü"))
        self.actionNeu.setText(_translate("MainWindow", "Neue Kompression..."))
        self.actionNeu.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionSchliessen.setText(_translate("MainWindow", "Schließen"))
        self.actionSchliessen.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionQuelle.setText(_translate("MainWindow", "Quelle Anzeigen"))
        self.actionQuelle.setShortcut(_translate("MainWindow", "Ctrl+O"))


class JPEGThread(QThread):

    signal = pyqtSignal()

    def __init__(self, filename):
        QThread.__init__(self)
        self.filename = filename

    def __del__(self):
        self.wait()

    def run(self):
        global wertemap
        global jpg_pfad
        jpg_pfad, ergebnisse = nvm.komprimiere(self.filename)

        wertemap = []
        for i in range(len(ergebnisse)):
            ergebnis = ergebnisse[i]
            if isinstance(ergebnis, list):
                wertemap.append(ergebnis[0])
                wertemap.append(ergebnis[1])
                wertemap.append(ergebnis[2])
            else:
                wertemap.append(ergebnis)
                wertemap.append(ergebnis[:,:,0])
                wertemap.append(ergebnis[:,:,1])
                wertemap.append(ergebnis[:,:,2])

        wertemap.append([ergebnisse[0], ergebnisse[len(ergebnisse) - 6]])
        wertemap.append([ergebnisse[0], ergebnisse[len(ergebnisse) - 1]])


        self.signal.emit()


class LoadThread(QThread):

    signal = pyqtSignal()

    def __init__(self, index, popup):
        QThread.__init__(self)
        self.popup = popup
        self.index = index

    def __del__(self):
        self.wait()

    def run(self):

        global qll
        global entsg
        global entr
        global mse
        global groesse1
        global groesse2
        global psnr
        global compfak

        if index < len(wertemap) - 2:
            matrix = wertemap[self.index]
            qll = nvm.quellenredundanz(matrix)
            entsg = nvm.entscheidungsgehalt(matrix)
            entr = nvm.entropie(matrix)
            groesse1 = nvm.groesse(matrix)

        else:
            matrix1 = wertemap[self.index][0]
            matrix2 = wertemap[self.index][1]
            mse = nvm.mse(matrix1, matrix2)
            groesse1 = nvm.groesse(matrix1)
            groesse2 = nvm.groesse(matrix2)
            psnr = nvm.psnr(matrix1, matrix2)
            compfak = nvm.compFak(groesse1, groesse2)

        self.signal.emit()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
