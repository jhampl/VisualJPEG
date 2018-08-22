# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter
import matplotlib.pyplot as plt
import os
import seaborn as davi
import time

np.set_printoptions(threshold=np.nan)
ordner = 'Ergebnisse'


class JPEG:
    def __init__(self, img):
        self.img = np.array(Image.open(img))
        self.img_path = img

        if not os.path.exists(ordner):
                os.makedirs(ordner)

        # TODO add downsampling
        self.rgb = self.crop()
        self.ycbcr = self.hinTransformation(self.rgb)
        self.dct = self.dct(self.ycbcr)
        self.dct_quant = self.quant(self.dct)
        self.dct_dequant = self.dequant(self.dct_quant)
        self.idct = self.inversedct(self.dct_dequant)
        self.jpg = self.rueckTransformation(self.idct)

        self.rgb_pfad = drucke_bild('rgb', self.rgb)
        self.jpg_pfad = drucke_bild('jpg', self.jpg)

        self.werte = [
            self.rgb, self.rgb[:, :, 0], self.rgb[:, :, 1],
            self.rgb[:, :, 2], self.ycbcr[:, :, 0], self.ycbcr[:, :, 1],
            self.ycbcr[:, :, 2], self.dct[:, :, 0], self.dct[:, :, 1],
            self.dct[:, :, 2], self.dct_quant[:, :, 0],
            self.dct_quant[:, :, 1], self.dct_quant[:, :, 2],
            self.dct_dequant[:, :, 0], self.dct_dequant[:, :, 1],
            self.dct_dequant[:, :, 2], self.idct[:, :, 0], self.idct[:, :, 1],
            self.idct[:, :, 2], self.jpg, self.jpg[:, :, 0], self.jpg[:, :, 1],
            self.jpg[:, :, 2] 
        ]

    def hinTransformation(self, matrix):
        nmatrix = matrix.copy()
        ycbcr = np.array([[.299, .587, .114], [-.169, -.331, .5],
                          [.5, -.419, -.081]])
        for a in range(nmatrix.shape[0]):
            for b in range(nmatrix.shape[1]):
                np.matmul(
                    ycbcr,
                    np.array(
                        [nmatrix[a][b][0], nmatrix[a][b][1], nmatrix[a][b][2]]),
                    nmatrix[a][b])
        nmatrix[:, :, [1, 2]] += 128
        return np.round(nmatrix)

    def rueckTransformation(self, matrix):
        nmatrix = matrix.copy()
        ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
        nmatrix[:, :, [1, 2]] -= 128

        for a in range(nmatrix.shape[0]):
            for b in range(nmatrix.shape[1]):
                np.matmul(
                    ycbcr,
                    np.array(
                        [nmatrix[a][b][0], nmatrix[a][b][1], nmatrix[a][b][2]]),
                    nmatrix[a][b])

        np.putmask(nmatrix, nmatrix > 255, 255)
        np.putmask(nmatrix, nmatrix < 0, 0)

        return nmatrix

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

    def dct(self, matrix):

        helpMat = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[2]))

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for z in range(0, matrix.shape[2]):
                    for u in range(0, 8):
                        for v in range(0, 8):

                            help = 0

                            for x in range(0, 8):
                                for y in range(0, 8):
                                    help += matrix[a + x][b + y][z] * math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                            math.cos(((2 * y + 1) * v * math.pi) / 16)

                            if u == 0:
                                cu = 1 / math.sqrt(2)
                            else:
                                cu = 1
                            if v == 0:
                                cv = 1 / math.sqrt(2)
                            else:
                                cv = 1

                            help = help * 1 / 4 * cu * cv

                            helpMat[a + u][b + v][z] = help

        return np.array(helpMat)

    def quant(self, matrix):

        help = np.zeros((512, 512, 3))

        QY = np.array([[16, 11, 10, 16, 24, 40, 51,
                        61], [12, 12, 14, 19, 26, 48, 60,
                              55], [14, 13, 16, 24, 40, 57, 69,
                                    56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103,
                        77], [24, 35, 55, 64, 81, 104, 113,
                              92], [49, 64, 78, 87, 103, 121, 120,
                                    101], [72, 92, 95, 98, 112, 100, 103, 99]])

        QC = np.array([[17, 18, 24, 47, 99, 99, 99,
                        99], [18, 21, 26, 66, 99, 99, 99,
                              99], [24, 26, 56, 99, 99, 99, 99,
                                    99], [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99,
                        99], [99, 99, 99, 99, 99, 99, 99,
                              99], [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

        for x in range(0, matrix.shape[0], 8):
            for y in range(0, matrix.shape[1], 8):
                for z in range(matrix.shape[2]):
                    for u in range(0, 8):
                        for v in range(0, 8):

                            if z == 0:
                                help[x + u][y + v][z] = int(
                                    round(matrix[x + u][y + v][z] / QY[u][v]))
                            else:
                                help[x + u][y + v][z] = int(
                                    round(matrix[x + u][y + v][z] / QC[u][v]))

        return np.array(help)

    def dequant(self, matrix):
        QY = np.array([[16, 11, 10, 16, 24, 40, 51,
                        61], [12, 12, 14, 19, 26, 48, 60,
                              55], [14, 13, 16, 24, 40, 57, 69,
                                    56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103,
                        77], [24, 35, 55, 64, 81, 104, 113,
                              92], [49, 64, 78, 87, 103, 121, 120,
                                    101], [72, 92, 95, 98, 112, 100, 103, 99]])

        QC = np.array([[17, 18, 24, 47, 99, 99, 99,
                        99], [18, 21, 26, 66, 99, 99, 99,
                              99], [24, 26, 56, 99, 99, 99, 99,
                                    99], [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99,
                        99], [99, 99, 99, 99, 99, 99, 99,
                              99], [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for z in range(0, 3):
                    for x in range(0, 8):
                        for y in range(0, 8):
                            if z == 0:
                                matrix[a + x][b + y][
                                    z] = matrix[a + x][b + y][z] * QY[x][y]
                            else:
                                matrix[a + x][b + y][
                                    z] = matrix[a + x][b + y][z] * QC[x][y]
        return matrix

    def inversedct(self, matrix):
        nmatrix = matrix.copy()
        helpMat = np.zeros((8, 8, 1))

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for z in range(0, 3):
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

                                    help += cu * cv * nmatrix[a +
                                                             u][b +
                                                                v][z] * math.cos(
                                                                    ((2 * x +
                                                                      1) * u *
                                                                     math.pi) /
                                                                    16
                                                                ) * math.cos((
                                                                    (2 * y + 1)
                                                                    * v * math.
                                                                    pi) / 16)

                            helpMat[x][y] = help * 1 / 4

                    for x in range(0, 8):
                        for y in range(0, 8):
                            nmatrix[a + x][b + y][z] = helpMat[x][y]

        return nmatrix

    def entropie(self, matrix):

        counts = Counter()

        entropy = 0
        i = 0

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if len(matrix.shape) == 3:
                    for z in range(matrix.shape[2]):
                        counts[matrix[x][y][z]] += 1
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
                        counts[matrix[x][y][z]] += 1
                else:
                    counts[matrix[x][y]] += 1

        lan = len(counts.keys())

        infogehalt = math.log(lan)

        return infogehalt

    def quellenredundanz(self, matrix):

        return (self.entropie(matrix) - self.entscheidungsgehalt(matrix))

    def MSE(self):
        return self.MSE_(self.rgb, self.jpg)

    def MSE_(self, Y, YH):
        return np.square(Y - YH).mean()

    def PSNR(self):
        return self.PSNR_(self.rgb, self.jpg)

    def PSNR_(self, Y, YH):
        max_val = 255
        mse = self.MSE_(Y, YH)
        psnr = 20 * math.log(max_val, 10) - 10 * math.log(mse, 10)

        return psnr

    def groessen(self):
        rgb_groesse = os.path.getsize(self.rgb_pfad)
        jpg_groesse = os.path.getsize(self.jpg_pfad)

        return rgb_groesse, jpg_groesse;

    def compFak(self):
        rgb_groesse, jpg_groesse = self.groessen()
        return (jpg_groesse / rgb_groesse)


    def compFak_(self, a, b):

        a_groesse = os.path.getsize(a)
        b_groesse = os.path.getsize(b)
        return (b_groesse / a_groesse)

    def save(self, matrix, path):

        pic = Image.fromarray(np.uint8(matrix))
        pic.save(path)

    def speichere_histogramm( self, label,  hist):
        bildpfad = pfad(label + '.png')
        fig.savefig(bildpfad)
        return bildpfad


    def zeige_histogramm(self, label, array):

        davi.set_style('whitegrid')
        eindim = np.reshape(array, -1)
        plot = davi.distplot(eindim)
        fig = plot.get_figure()
        plt.show()
        # fig.savefig(pfad(label + '.png'))
        plt.close(fig)


def pfad(datei):

    return os.path.abspath( ordner + '/' + datei)

def drucke_bild( label, array):

    img = Image.fromarray(array.astype('uint8'))
    bildpfad = pfad(label + '.png')
    img.save(bildpfad)
    return bildpfad


def histogramm(array):

    plt.close('all')
    davi.set_style('whitegrid')

    eindim = np.reshape(array, -1)

    plot = davi.distplot(eindim)
    fig= plot.get_figure()

    return fig

if __name__ == '__main__':
    pass
