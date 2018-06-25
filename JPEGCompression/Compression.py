# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from PIL import Image
import math

np.set_printoptions(threshold=np.nan)

class JPEG:

    def __init__(self, img):
        self.img = np.array(Image.open(img))

    def openShow(self):
        img = Image.open(self.img)
        Image._show(img)

    def rgb2ycbcr(self, matrix):
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = matrix.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.round(ycbcr)

    def ycbcr2rgb(self, im):
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = im.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return np.round(rgb)

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
        return np.uint8(np.round(matrix))

    def showYCBCR(self):
        img = Image.fromarray(self.rgb2ycbcr())
        Image._show(img)

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

    def getImage(self):
        image = self.img
        return image

    def partition(self, block_size):
        x = self.crop(8)

        data = np.split(x, (x.shape[0] / block_size)) #512x512 : 64, wenn 8 -> 64x64 mit jeweils 8
        res = []


        for number, arr in enumerate(data):
            res.append(np.split(arr, arr.shape[1] / block_size, axis=1))
        return res

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

        # blöcke
        for a in range(0, matrix.shape[0], 8):  # start, stop, step
            for b in range(0, matrix.shape[1], 8):

                # drei ebenen
                for z in range(0,3):
                    #print('---------')

                    # äußere schleife
                    for u in range(0, 8):
                        for v in range(0, 8):
                            #print(matrix[a+u][b+v][z])

                            help = 0

                            #innere schleife
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

        for a in range(0, matrix.shape[0], 8):  # start, stop, step
            for b in range(0, matrix.shape[1], 8):

                # drei ebenen
                for z in range(0,3):

                    # äußere schleife
                    for x in range(0, 8):
                        for y in range(0, 8):
                            if z == 0:
                                matrix[a+x][b+y][z] = matrix[a+x][b+y][z]*QY[x][y]
                            else:
                                matrix[a+x][b+y][z] = matrix[a+x][b+y][z]*QC[x][y]
        return matrix


    def inversedct(self, matrix):
        helpMat = np.zeros((8, 8, 1))

        # blöcke
        for a in range(0, matrix.shape[0], 8):  # start, stop, step
            for b in range(0, matrix.shape[1], 8):

                # drei ebenen
                for z in range(0,3):

                    # äußere schleife
                    for x in range(0, 8):
                        for y in range(0, 8):

                            help = 0

                            #innere schleife
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

                                    help += round(cu * cv * matrix[a+u][b+v][z] * math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                                                            math.cos(((2 * y + 1) * v * math.pi) / 16))

                            helpMat[x][y] = int(round(help * 1/4))

                    for x in range(0, 8):
                        for y in range(0, 8):
                            matrix[a+x][b+y][z] = helpMat[x][y]
        return matrix


if __name__ == '__main__':

    nvm = JPEG('/Users/jow/Dropbox/Forensik/Master/2.Semester/Datenkompression/Vorlesung/Testbilder/Lena/4.2.04.png')
    #res = nvm.getImage()
    res = nvm.crop()

    #res = nvm.dct(res)
    res = nvm.hinTransformation(res)

    res = nvm.dct(res)

    for i in range(8):
        for j in range(8):
            print(res[8*0+i][8*0+j][0])

    res = nvm.dequant(res)
    res = nvm.inversedct(res)

    print('--------')

    for i in range(8):
        for j in range(8):
            print(res[8 * 0 + i][8 * 0 + j][0])

    res = nvm.rueckTransformation(res)

    #res[:, :, [0,2]] = 0

    nvm.showState(res)
