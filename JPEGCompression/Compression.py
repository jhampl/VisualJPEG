# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter
import seaborn as davi
from shutil import copyfile
import os
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

lRGB, lYCbCr, lYCbCr_sub, lDCT, lDCT_quant, lDCT_dequant, lIDCT, lYCbCr_over, lJPG = 'RGB', 'YCbCr', 'YCbCr_sub', 'DCT', 'DCT_quant', 'lDCT_dequant', 'lIDCT', 'lYCbCr_over', 'JPG' 
labels = [lRGB, lYCbCr, lYCbCr_sub, lDCT, lDCT_quant, lDCT_dequant, lIDCT, lYCbCr_over, lJPG]
lR, lG, lB, lY, lCb, lCr = 'R', 'G', 'B', 'Y', 'Cb', 'Cr'

class JPEG:
        
    def __init__(self, img):
        self.img = np.array(Image.open(img))
        self.ordner = 'Ergebnisse'

        self.rgb = self.crop() 
        self.ycbcr = self.hinTransformation(self.rgb)
        self.sub = self.unterabtastung(self.ycbcr)
        self.dct = self.dct(self.sub)
        self.dct_quant = self.quant(self.dct)
        self.dct_dequant = self.dequant(self.dct_quant)
        self.idct = self.inversedct(self.dct_dequant)
        self.ueb = self.ueberabstastung(self.idct)
        self.jpg = self.rueckTransformation(self.ueb)

        self.schritte = [ rgb, ycbcr, sub, dct, dct_quant, dct_dequant, idct, ueb, jpg ] 


    def hinTransformation(self, matrix):
        ycbcr = np.array(
            [[.299, .587, .114], [-.169, -.331, .5], [.5, -.419, -.081]])
        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(ycbcr, np.array(
                    [matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]), matrix[a][b])
        matrix[:, :, [1, 2]] += 128
        return np.round(matrix)

    def unterabtastung(self, matrix):
        y = matrix[:,:,0]
        cb = matrix[::2,::2,1]
        cr = matrix[::2,::2,2]
        
        return y, cb, cr


    def ueberabstastung(self,matrix):
        y = matrix[0]
        cb = matrix[1]
        cr = matrix[2]
        interp = np.zeros((y.shape[0], y.shape[1], 3))

        interp[:,:,0] = y
        interp[:,:,1] = cb.repeat(2, axis = 0).repeat(2, axis = 1)
        interp[:,:,2] = cr.repeat(2, axis = 0).repeat(2, axis = 1)

        return interp


    def rueckTransformation(self, matrix):
        ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
        matrix[:, :, [1, 2]] -= 128

        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(ycbcr, np.array(
                    [matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]), matrix[a][b])

        np.putmask(matrix, matrix > 255, 255)
        np.putmask(matrix, matrix < 0, 0)

        return matrix


    def showState(self, matrix):
        im = Image.fromarray(np.uint8(matrix))
        Image._show(im)


    def crop(self):
        image = self.img

        if image.shape[0] % 8 != 0:
            cropped_width = math.floor(int(image.shape[0] / 8) * 8)
        else:
            cropped_width = image.shape[0]

        if image.shape[1] % 8 != 0:
            cropped_height = math.floor(int(image.shape[1] / 8) * 8)
        else:
            cropped_height = image.shape[1]

        cropped_image = np.zeros((cropped_width, cropped_height, 3))

        for rownum in range(len(cropped_image)):
            for colnum in range(len(cropped_image[rownum])):
                cropped_image[rownum][colnum] = image[rownum][colnum]
        return cropped_image


    def dct(self, matrizen):

        for z in range(0, len(matrizen)):
            matrix = matrizen[z]
            helpMat = np.zeros((matrix.shape[0], matrix.shape[1]))
            for a in range(0, matrix.shape[0], 8):
                for b in range(0, matrix.shape[1], 8):
                    for u in range(0, 8):
                        for v in range(0, 8):

                            help = 0

                            for x in range(0, 8):
                                for y in range(0, 8):
                                    help += matrix[a+x][b+y] * math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                        math.cos(
                                            ((2 * y + 1) * v * math.pi) / 16)

                            if u == 0:
                                cu = 1 / math.sqrt(2)
                            else:
                                cu = 1
                            if v == 0:
                                cv = 1 / math.sqrt(2)
                            else:
                                cv = 1

                            help = help * 1/4 * cu * cv

                            # matrix[a+u][b+v][z] = help

                        for r in range(0, 8):
                            for s in range(0, 8):
                                matrix[a+r][b+s] = helpMat[r][s]
        return matrizen


    def quant(self, matrizen):

        QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 48, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

        QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])
        
        for z in range(0, len(matrizen)):
            matrix = matrizen[z]
            helpMat = np.zeros((matrix.shape[0], matrix.shape[1]))
            for a in range(0, matrix.shape[0], 8):
                for b in range(0, matrix.shape[1], 8):
                    for x in range(0, 8):
                        for y in range(0, 8):
                            if z == 0:
                                helpMat[u][v] = int(round(help/QY[u][v]))
                            else:
                                helpMat[u][v] = int(round(help/QC[u][v]))
        return matrizen


    def dctquant(self, matrizen):
        QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 48, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

        QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

        helpMat = np.zeros((8, 8, 1))

        for z in range(0, len(matrizen)):
            matrix = matrizen[z]
            for a in range(0, matrix.shape[0], 8):
                for b in range(0, matrix.shape[1], 8):
                    for u in range(0, 8):
                        for v in range(0, 8):

                            help = 0

                            for x in range(0, 8):
                                for y in range(0, 8):
                                    help += matrix[a+x][b+y] * math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                        math.cos(
                                            ((2 * y + 1) * v * math.pi) / 16)

                            if u == 0:
                                cu = 1 / math.sqrt(2)
                            else:
                                cu = 1
                            if v == 0:
                                cv = 1 / math.sqrt(2)
                            else:
                                cv = 1

                            help = help * 1/4 * cu * cv

                            # matrix[a+u][b+v][z] = help

                            # print(help)
                            if z == 0:
                                helpMat[u][v] = int(round(help/QY[u][v]))
                            else:
                                helpMat[u][v] = int(round(help/QC[u][v]))

                    for r in range(0, 8):
                        for s in range(0, 8):
                            matrix[a+r][b+s] = helpMat[r][s]
        return matrizen


    def dequant(self, matrizen):
        QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 48, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

        QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

        for z in range(0, len(matrizen)):
            matrix = matrizen[z]
            for a in range(0, matrix.shape[0], 8):
                for b in range(0, matrix.shape[1], 8):
                    for x in range(0, 8):
                        for y in range(0, 8):
                            if z == 0:
                                matrix[a+x][b+y] = matrix[a +
                                                             x][b+y]*QY[x][y]
                            else:
                                matrix[a+x][b+y] = matrix[a +
                                                             x][b+y]*QC[x][y]
        return matrizen


    def inversedct(self, matrizen):
        helpMat = np.zeros((8, 8, 1))

        for z in range(0, len(matrizen)): 
            matrix = matrizen[z]
            for a in range(0, matrix.shape[0], 8):
                for b in range(0, matrix.shape[1], 8):
                    for x in range(0, 8):
                        for y in range(0, 8):

                            help = 0

                            for u in range(0, 8):
                                for v in range(0, 8):
                                    if u == 0:
                                        cu = 1 / math.sqrt(2)
                                    else:
                                        cu = 1
                                    if v == 0:
                                        cv = 1 / math.sqrt(2)
                                    else:
                                        cv = 1

                                    help += cu * cv * matrix[a+u][b+v] * math.cos(
                                        ((2 * x + 1) * u * math.pi) / 16) * math.cos(((2 * y + 1) * v * math.pi) / 16)

                            helpMat[x][y] = help * 1/4

                        for x in range(0, 8):
                            for y in range(0, 8):
                                matrix[a+x][b+y] = helpMat[x][y]

        return matrizen


    def entropie(self, matrix):

        counts = Counter()

        entropy = 0
        i = 0

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if len(matrix.shape) == 3:
                    for z in range(matrix.shape[2]):
                        i += 1
                else:
                    counts[matrix[x][y]] += 1
                    i += 1

        probs = [float(c) / i for c in counts.values()]

        for p in probs:
            if p > 0.:
                entropy -= p * math.log(p, 2)

        return entropy


    def entscheidungsgehalt(self, matrix):

        counts = Counter()

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if len(matrix.shape) == 3:
                    for z in range(matrix.shape[2]):
                        counts[matrix[x][y]] += 1
                else:
                    counts[matrix[x][y]] += 1

        lan = len(counts.keys())

        infogehalt = math.log(lan)

        return infogehalt


    def quellenredundanz(self, matrix):

        return (nvm.entropie(matrix) - nvm.entscheidungsgehalt(matrix))


    def drucke_histogramm(self, label, array):

        davi.set_style('whitegrid')
        eindim = np.reshape(array, -1)
        # print(eindim)
        plot = davi.distplot(eindim)
        fig = plot.get_figure()
        # plt.show()
        fig.savefig(self.pfad(label + '.png'))
        plt.close(fig)


    def zeige_histogramm(self, label, array):

        davi.set_style('whitegrid')
        eindim = np.reshape(array, -1)
        # print(eindim)
        plot = davi.distplot(eindim)
        fig = plot.get_figure()
        plt.show()
        # fig.savefig(self.pfad(label + '.png'))
        plt.close(fig)


    def drucke_bild(self, label, array):

        img = Image.fromarray(array, 'RGB')
        img.save(self.pfad(label + '.png'))


    def pfad(self, datei):

        return os.path.abspath(self.ordner + '/' + datei)


# if __name__ == '__main__':

    # # Initialisierung
    # src = os.path.abspath('../Testbilder/Lena/4.2.04.png')
    # nvm = JPEG(src)

    # # Ordner erstellen
    # if not os.path.exists(wurzel):
        # os.makedirs(wurzel)

    # for ordner in labels:
        # os.makedirs(wurzel + '/' + ordner)

   # # RGB Bild in Ergebnissen speichern
    # copyfile(src, nvm.pfad('rgb.png'))

    # rgb = nvm.crop() 
    # ycbcr = nvm.hinTransformation(rgb)
    # sub = nvm.unterabtastung(ycbcr)
    # dct = nvm.dct(sub)
    # dct_quant = nvm.quant(dct)
    # dct_dequant = nvm.dequant(dct_quant)
    # idct = nvm.inversedct(dct_dequant)
    # ueb = nvm.ueberabstastung(idct)
    # jpg = nvm.rueckTransformation(ueb)

    # schritte = [ rgb, ycbcr, sub, dct, dct_quant, dct_dequant, idct, ueb, jpg ] 
    
    # for s in 0:len(schritte):

        # ordner = wurzel + labels[s] 

        # if 0 < counter < 8:
            # header = [ 'Gesamt', 'R', 'G', 'B' ]

        # else:
            # header = [ 'Gesamt', 'Y', 'Cb', 'Cr' ]
            
        # drucke_bild(label, matrix)
        # drucke_histogramm


    # # RGB Histogramme erstellen
    # nvm.drucke_histogramm('R', nvm.img[:, :, 0])
    # nvm.drucke_histogramm('G', nvm.img[:, :, 1])
    # nvm.drucke_histogramm('B', nvm.img[:, :, 2])

    # # Farbraumtransformation
    # result = nvm.crop()
    # result = nvm.hinTransformation(result)

    # # YCbCr Histogramme erstellen
    # nvm.drucke_histogramm('Y', result[:, :, 0])
    # nvm.drucke_histogramm('Cb', result[:, :, 1])
    # nvm.drucke_histogramm('Cr', result[:, :, 2])

    # # Maße errechnen und schreiben
    # # RGB Maße
    # header = 'Komponente\tEntropie\tEntscheidungsgehalt\tQuellenredundanz'
    # with open(nvm.pfad('RGB_daten.txt'), 'w+') as daten:
        # daten.write(header)

    # komps = 'Gesamt', 'Y', 'Cb', 'Cr'
    # komp = '' 
    # entrp = '' 
    # entschg = '' 
    # qllred = ''

    # for i in [ -1, 0, 1, 2 ]:
        # if i == -1:
            # komp = komps[0]
            # entrp = nvm.entropie(result[:, :, :])
            # entschg = nvm.entscheidungsgehalt(result[:, :, :])
            # qllred = nvm.quellenredundanz(result[:, :, :])

        # else:
            # komp = komps[i+1]
            # entrp = nvm.entropie(result[:, :, i])
            # entschg = nvm.entscheidungsgehalt(result[:, :, i])
            # qllred = nvm.quellenredundanz(result[:, :, i])

        # line = f'\n{komp}\t{entrp}\t{entschg}\t{qllred}'
        # with open(nvm.pfad('RGB_daten.txt'), 'a+') as daten:
            # daten.write(line)

    # # print nvm.entropie(result[:,:,0])
    # # print nvm.entscheidungsgehalt(result[:,:,0])
    # # print nvm.quellenredundanz(result[:,:,0])

    # result = nvm.dct(result)

    # # print nvm.entropie(result)
    # # print nvm.entscheidungsgehalt(result)
    # # print nvm.quellenredundanz(result)
