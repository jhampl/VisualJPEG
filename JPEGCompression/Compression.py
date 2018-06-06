# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from PIL import Image
import math

class JPEG:

    def __init__(self, img):
        self.img = np.array(Image.open(img))

    def openShow(self):
        img = Image.open(self.img)
        Image._show(img)

    def rgb2ycbcr(self):
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = self.img.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)

    def showYCBCR(self):
        img = Image.fromarray(self.rgb2ycbcr())
        Image._show(img)

    def crop(self, block_size):
        image = self.img

        if image.shape[0] % block_size != 0:
            cropped_width = math.floor(int(image.shape[0] / block_size) * block_size)
        else:
            cropped_width = image.shape[0]

        if image.shape[1] % block_size != 0:
            cropped_height = math.floor(int(image.shape[1] / block_size) * block_size)
        else:
            cropped_height = image.shape[1]

        cropped_image = np.zeros((cropped_width, cropped_height,3))

        for rownum in range(len(cropped_image)):
            for colnum in range(len(cropped_image[rownum])):
                cropped_image[rownum][colnum] = image[rownum][colnum]

        return cropped_image

    def partition(self, block_size):
        x = np.array(self.img)
        data = np.split(x, x.shape[0] / block_size) #512x512 : 64, wenn 8 -> 64x64 mit jeweils 8
        res = []

        for number, arr in enumerate(data):
            res.append(np.split(arr, arr.shape[1] / block_size, axis=1))

        return res

if __name__ == '__main__':

    nvm = JPEG('/Users/Paul/Desktop/4.2.04.png')
    nvm.crop(8)