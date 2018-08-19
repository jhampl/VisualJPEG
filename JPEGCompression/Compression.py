# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter
import matplotlib.pyplot as plt
import os
import seaborn as davi

np.set_printoptions(threshold=np.nan)


class JPEG:
    def __init__(self, img):
        self.img = np.array(Image.open(img))
        self.img_path = img

        # TODO add downsampling
        self.rgb = self.crop()
        self.ycbcr = self.hinTransformation(self.rgb)
        self.dct = self.dct(self.ycbcr)
        self.dct_quant = self.quant(self.dct)
        self.dct_dequant = self.dequant(self.dct_quant)
        self.idct = self.inversedct(self.dct_dequant)
        self.jpg = self.rueckTransformation(self.idct)

        self.rgb_pfad = self.drucke_bild('rgb', self.rgb)
        self.jpg_pfad = self.drucke_bild('jpg', self.jpg)

        self.werte = [
            self.rgb, self.ycbcr[:, :, 0], self.ycbcr[:, :, 1],
            self.ycbcr[:, :, 2], self.dct[:, :, 0], self.dct[:, :, 1],
            self.dct[:, :, 2], self.dct_quant[:, :, 0],
            self.dct_quant[:, :, 1], self.dct_quant[:, :, 2],
            self.dct_dequant[:, :, 0], self.dct_dequant[:, :, 1],
            self.dct_dequant[:, :, 2], self.idct[:, :, 0], self.idct[:, :, 1],
            self.idct[:, :, 2], self.jpg
        ]

    def hinTransformation(self, matrix):
        ycbcr = np.array([[.299, .587, .114], [-.169, -.331, .5],
                          [.5, -.419, -.081]])
        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(
                    ycbcr,
                    np.array(
                        [matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]),
                    matrix[a][b])
        matrix[:, :, [1, 2]] += 128
        return np.round(matrix)

    def rueckTransformation(self, matrix):
        ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
        matrix[:, :, [1, 2]] -= 128

        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(
                    ycbcr,
                    np.array(
                        [matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]),
                    matrix[a][b])

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

                                    help += cu * cv * matrix[a +
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
                            matrix[a + x][b + y][z] = helpMat[x][y]

        return matrix

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

    def histogram(self, matrix, title):

        array = []

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if len(matrix.shape) == 2:
                    array.append(matrix[x][y])
                else:
                    for z in range(matrix.shape[2]):
                        array.append(matrix[x][y][z])

        plt.hist(array, bins='auto')
        plt.title("Histogram" + " " + title)
        plt.show()

    def MSE(self):
        return self.MSE_(self.rgb, self.jpg)

    def MSE_(self, Y, YH):
        return np.square(Y - YH).mean()

    def PSNR(self):
        return self.PSNR_(self.rgb, self.jpg)

    def PSNR_(self, Y, YH):
        max_val = 255
        mse = self.MSE(Y, YH)
        psnr = 20 * math.log(max_val, 10) - 10 * math.log(mse, 10)

        return psnr

    def groessen():
        rgb_groesse = os.path.getsize(self.rgb_pfad)
        jpg_groesse = os.path.getsize(self.jpg_pfad)

        return rgb_groesse, jpg_groesse;

    def compFak(self):
        rgb_groesse, jpg_groesse = self.groessen()
        return self.compFak_(rgb_groesse, jpg_groesse)


    def compFak_(self, a, b):

        a_groesse = os.path.getsize(a)
        b_groesse = os.path.getsize(b)
        return (b_groesse / a_groesse)

    def save(self, matrix, path):

        pic = Image.fromarray(np.uint8(matrix))
        pic.save(path)

    def drucke_tmp_histogramm(self, array):

        return self.drucke_histogramm('tmphist', array)
    
    def drucke_histogramm(self, label, array):

        davi.set_style('whitegrid')
        eindim = np.reshape(array, -1)
        bildpfad = self.pfad(label + '.png')

        if len(array.shape) == 3:
            figs = [ davi.distplot(np.reshape(array[:,:,0], -1)).get_figure(), 
                    davi.distplot(np.reshape(array[:,:,1], -1)).get_figure(), 
                    davi.distplot(np.reshape(array[:,:,2], -1)).get_figure() ]
            fig = figs[2]
            fig.savefig(bildpfad)
            for fig in figs:
                plt.close(fig)
        
        else:
            # print(eindim)
            plot = davi.distplot(eindim)
            fig = plot.get_figure()
            # plt.show()
            fig.savefig(bildpfad)
            plt.close(fig)
        
        return bildpfad

    def zeige_histogramm(self, label, array):

        davi.set_style('whitegrid')
        eindim = np.reshape(array, -1)
        # print(eindim)
        plot = davi.distplot(eindim)
        fig = plot.get_figure()
        plt.show()
        # fig.savefig(self.pfad(label + '.png'))
        plt.close(fig)

    def drucke_tmp_bild(self, array):

        return self.drucke_bild('tmp', array)

    def drucke_bild(self, label, array):

        img = Image.fromarray(array, 'RGB')
        bildpfad = self.pfad(label + '.png')
        img.save(bildpfad)
        return bildpfad

    def pfad(self, datei):

        return os.path.abspath('./' + datei)

if __name__ == '__main__':

    nvm = JPEG(
        'I:\misc/4.2.04.png')  # Instanz - Bild wird Konstruktor übergeben
