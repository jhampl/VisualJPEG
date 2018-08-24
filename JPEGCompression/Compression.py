# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter
import matplotlib.pyplot as plt
from itertools import groupby
import os

np.set_printoptions(threshold=np.nan)


class JPEG:


    def __init__(self, img):

        self.img = np.array(Image.open(img))

    def hinTransformation(self, matrix):

        ycbcr = np.array([[.299, .587, .114], [-.169, -.331, .5], [.5, -.419, -.081]])
        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(ycbcr, np.array([matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]), matrix[a][b])
        matrix[:, :, [1, 2]] += 128
        return np.round(matrix)

    def rueckTransformation(self, matrix):

        ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
        matrix[:, :, [1, 2]] -= 128

        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(ycbcr, np.array([matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]), matrix[a][b])

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

    def pad(self, image):
        if image.shape[0] % 8 == image.shape[1] % 8 == 0:
            return image
        y_pad = 8 - image.shape[0] % 8
        x_pad = 8 - image.shape[1] % 8

        return np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), 'reflect')

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
                                    help += matrix[a + x][b + y][z] * \
                                            math.cos(((2 * x + 1) * u * math.pi) / 16) * \
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

        help = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[2]))

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

        for x in range(0, matrix.shape[0], 8):
            for y in range(0, matrix.shape[1], 8):
                for z in range(matrix.shape[2]):
                    for u in range(0, 8):
                        for v in range(0, 8):

                            if z == 0:
                                help[x + u][y + v][z] = int(round(matrix[x + u][y + v][z] / QY[u][v]))
                            else:
                                help[x + u][y + v][z] = int(round(matrix[x + u][y + v][z] / QC[u][v]))

        return np.array(help)

    def dequant(self, matrix):

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

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for z in range(0, 3):
                    for x in range(0, 8):
                        for y in range(0, 8):
                            if z == 0:
                                matrix[a + x][b + y][z] = matrix[a + x][b + y][z] * QY[x][y]
                            else:
                                matrix[a + x][b + y][z] = matrix[a + x][b + y][z] * QC[x][y]
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

                                    help += cu * cv * matrix[a + u][b + v][z] * \
                                            math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                            math.cos(((2 * y + 1) * v * math.pi) / 16)

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
        infogehalt = math.log(lan, 2)

        return infogehalt

    def quellenredundanz(self, matrix):

        return (nvm.entropie(matrix) - nvm.entscheidungsgehalt(matrix))

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

    def MSE(self, Y, YH):

        return np.square(Y - YH).mean()

    def PSNR(self, Y, YH):

        max_val = 255
        mse = self.MSE(Y, YH)
        psnr = 20 * math.log(max_val, 10) - 10 * math.log(mse, 10)

        return psnr

    def compFak(self, Y, YH):

        beginn = os.path.getsize(Y)
        ende = os.path.getsize(YH)

        return (ende / beginn)

    def save(self, matrix, path):

        pic = Image.fromarray(np.uint8(matrix))
        pic.save(path)

    def runEncode(table, matrix):

        array = []

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for x in range(np.array(table).shape[1]):
                    array.append(matrix[a + table[1][x]][b + table[0][x]])

        res = []

        for k, i in groupby(array):
            run = list(i)

            if len(run) > 3:
                res.append("({},{})".format(len(run), k))
            else:
                res.extend(run)

        return res

    def runDecode(table, array, size1, size2):

        alt = []

        matrix = np.zeros((size1, size2))

        for x in array:
            if type(x) is str:

                val = x.split(',')
                firstVal = str(val[0])[1:]
                secondVal = str(val[1])[0:len(str(val[1])) - 1]

                for number in range(int(firstVal)):
                    alt.append(int(secondVal))

            else:

                alt.append(x)

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for x in range(np.array(table).shape[1]):
                    matrix[a + table[1][x]][b + table[0][x]] = int(alt[x])

        return np.array(matrix)

    def main(self):
        pass


    def unterabtastung(self, matrix):
        return matrix[::2, ::2]


    def ueberabtastung(self, matrix):
        return matrix.repeat(2, axis=0).repeat(2, axis=1)


if __name__ == '__main__':
    nvm = JPEG('I:\misc/4.2.04.png')

    #table = [[0,1,0,0,1,2,3,2,1,0,0,1,2,3,4,5,4,3,2,1,0,0,1,2,3,4,5,6,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,7,6,5,4,3,2,3,4,5,6,7,7,6,5,4,5,6,7,7,6,7],
    #          [0,0,1,2,1,0,0,1,2,3,4,3,2,1,0,0,1,2,3,4,5,6,5,4,3,2,1,0,0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,2,3,4,5,6,7,7,6,5,4,3,4,5,6,7,7,6,5,6,7,7]]

    # ANWENDUNG FOLGT HIER ... nvm.main()