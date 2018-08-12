# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter

np.set_printoptions(threshold=np.nan)

class JPEG:

    def __init__(self, img):
        self.img = np.array(Image.open(img))


    def hinTransformation(self,matrix):
        ycbcr = np.array([[.299, .587, .114], [-.169, -.331, .5], [.5, -.419, -.081]])
        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(ycbcr,np.array([matrix[a][b][0],matrix[a][b][1],matrix[a][b][2]]), matrix[a][b])
        matrix[:, :, [1, 2]] += 128
        return np.round(matrix)


    def rueckTransformation(self,matrix):
        ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
        matrix[:, :, [1, 2]] -= 128

        for a in range(matrix.shape[0]):
            for b in range(matrix.shape[1]):
                np.matmul(ycbcr,np.array([matrix[a][b][0],matrix[a][b][1],matrix[a][b][2]]), matrix[a][b])

        np.putmask(matrix, matrix > 255, 255)
        np.putmask(matrix, matrix < 0, 0)

        return matrix


    def showState(self,matrix):
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

        cropped_image = np.zeros((cropped_width, cropped_height,3))

        for rownum in range(len(cropped_image)):
            for colnum in range(len(cropped_image[rownum])):
                cropped_image[rownum][colnum] = image[rownum][colnum]
        return cropped_image


    def dct(self, matrix):
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

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for z in range(0,3):
                    for u in range(0, 8):
                        for v in range(0, 8):

                            help = 0

                            for x in range(0,8):
                                for y in range(0,8):
                                    help += matrix[a+x][b+y][z] * math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                                                    math.cos(((2 * y + 1) * v * math.pi) / 16)

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

                            #print(help)
                            if z == 0:
                                helpMat[u][v] = int(round(help/QY[u][v]))
                            else:
                                helpMat[u][v] = int(round(help/QC[u][v]))

                    for r in range(0, 8):
                        for s in range(0, 8):
                            matrix[a+r][b+s][z] = helpMat[r][s]
        return matrix


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
                for z in range(0,3):
                    for x in range(0, 8):
                        for y in range(0, 8):
                            if z == 0:
                                matrix[a+x][b+y][z] = matrix[a+x][b+y][z]*QY[x][y]
                            else:
                                matrix[a+x][b+y][z] = matrix[a+x][b+y][z]*QC[x][y]
        return matrix


    def inversedct(self, matrix):
        helpMat = np.zeros((8, 8, 1))

        for a in range(0, matrix.shape[0], 8):
            for b in range(0, matrix.shape[1], 8):
                for z in range(0,3):
                    for x in range(0, 8):
                        for y in range(0, 8):

                            help = 0

                            for u in range(0,8):
                                for v in range(0,8):
                                    if u == 0:
                                        cu = 1 / math.sqrt(2)
                                    else:
                                        cu = 1
                                    if v == 0:
                                        cv = 1 / math.sqrt(2)
                                    else:
                                        cv = 1

                                    help += cu * cv * matrix[a+u][b+v][z] * math.cos(((2 * x + 1) * u * math.pi) / 16) * math.cos(((2 * y + 1) * v * math.pi) / 16)

                            helpMat[x][y] = help * 1/4

                    for x in range(0, 8):
                        for y in range(0, 8):
                            matrix[a+x][b+y][z] = helpMat[x][y]

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

        return (nvm.entropie(matrix) - nvm.entscheidungsgehalt(matrix))


if __name__ == '__main__':

    nvm = JPEG('/Users/Paul/Desktop/Master/2. Semester Master/Datenkompression/4.2.04.png')

    result = nvm.crop()
    result = nvm.hinTransformation(result)

    # [R][G][B] - [Y][Cb][Cr]

    print nvm.entropie(result[:,:,0])
    print nvm.entscheidungsgehalt(result[:,:,0])
    print nvm.quellenredundanz(result[:,:,0])

    result = nvm.dct(result)

    print nvm.entropie(result)
    print nvm.entscheidungsgehalt(result)
    print nvm.quellenredundanz(result)

