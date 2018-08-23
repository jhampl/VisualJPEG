# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter
import matplotlib.pyplot as plt
import os
import seaborn as davi
from multiprocessing import Pool
import sys
from itertools import groupby

np.set_printoptions(threshold=np.nan)
ordner = 'Ergebnisse'


def komprimiere(pfad):

    # Einlesen des Bilds
    img = np.array(Image.open(pfad))

    if not os.path.exists(ordner):
        os.makedirs(ordner)

    # Reflektieren des Bildrands fuer die 8x8 Blockbildung
    rgb = pad(img)

    # Transformation in YCbCr
    ycbcr = hinTransformation(rgb)

    # Paralleles Verarbeiten der Bildkomponenten,
    # siehe Methode "_komprimiereKomponent"
    pool = Pool(5)
    komponente = [[ycbcr[:,:,0], 0], [ycbcr[:,:,1], 1], [ycbcr[:,:,2], 2]]
    verarb_komponente = pool.map(_komprimiereKomponent, komponente)

    def reihe(i):
        return [verarb_komponente[0][i],
                verarb_komponente[1][i], verarb_komponente[2][i]]

    ycbcr_sub = reihe(0)
    dct = reihe(1)
    dct_quant = reihe(2)
    dct_cod = reihe(3)
    dct_decod = reihe(4)
    dct_dequant = reihe(5)
    idct = reihe(6)
    ycbcr_up = reihe(7)

    # Ruecktransformation in RGB
    jpg = rueckTransformation(ycbcr_up)

    rgb_pfad = drucke_bild('png', rgb)
    jpg_pfad = drucke_bild('jpg', jpg)

    ergebnisse = [rgb, ycbcr, ycbcr_sub, dct, dct_quant, dct_cod,
                  dct_decod, dct_dequant, idct, ycbcr_up, jpg]

    return jpg_pfad, ergebnisse


def _komprimiereKomponent(komponent):
    ycbcr = komponent[0].copy()
    chrominanz = komponent[1] > 0

    # Test ob Chrominanz-Komponent
    if chrominanz:
        # Unterabtasten der Chrominanz
        ycbcr_sub = unterabtastung(ycbcr)
    else:
        ycbcr_sub = ycbcr

    # Diskrete Cosinus Transformation in 8x8 Bloecken
    dct = dcTransformation(ycbcr_sub)
    # Quantisierung der Koeffizienten
    dct_quant = quant(dct, chrominanz)
    # Codierung der quantisierten Koeffizienten
    dct_cod = codierung(dct_quant)
    # Deodierung der Koeffizienten
    dct_decod = codierung(dct_cod)
    # Dequantisierung der Koeffizienten
    dct_dequant = dequant(dct_quant, chrominanz)
    # Inverse diskrete Cosinus Transformation in 8x8 Bloecken
    idct = idcTransformation(dct_dequant)

    # Test ob Chrominanz-Komponent
    if chrominanz:
        # Ueberabtastung der Chrominanz/Interpolation
        ycbcr_up = ueberabtastung(idct)
    else:
        ycbcr_up = idct

    return [ycbcr_sub, dct, dct_dequant, dct_cod,
            dct_decod, dct_quant, idct, ycbcr_up]


def hinTransformation(matrix):
    nmatrix = matrix.copy().astype(np.float64)
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
    nmatrix = np.round(nmatrix)

    return nmatrix


def rueckTransformation(komponente):

    matrix = np.zeros((komponente[0].shape[0], komponente[0].shape[1], 3))

    for i in range(3):
        matrix[:, :, i] = komponente[i]

    ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
    matrix[:, :, [1, 2]] -= 128

    for a in range(matrix.shape[0]):
        for b in range(matrix.shape[1]):
            np.matmul(ycbcr, np.array(
                [matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]), matrix[a][b])

        np.putmask(matrix, matrix > 255, 255)
        np.putmask(matrix, matrix < 0, 0)

    return matrix


def unterabtastung(matrix):
    return matrix[::2, ::2]


def ueberabtastung(matrix):
    return matrix.repeat(2, axis=0).repeat(2, axis=1)


def showState(matrix):
    im = Image.fromarray(np.uint8(matrix))
    Image._show(im)


def pad(image):
    if image.shape[0] % 8 == image.shape[1] % 8 == 0:
        return image
    y_pad = 8 - image.shape[0] % 8
    x_pad = 8 - image.shape[1] % 8

    return np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), 'reflect')


def dcTransformation(matrix):

    helpMat = np.zeros((matrix.shape[0], matrix.shape[1]))

    for a in range(0, matrix.shape[0], 8):
        for b in range(0, matrix.shape[1], 8):
            for u in range(0, 8):
                for v in range(0, 8):

                    help = 0

                    for x in range(0, 8):
                        for y in range(0, 8):
                            help += matrix[a + x][b + y] *                  \
                                math.cos(((2 * x + 1) * u * math.pi) / 16) *\
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

                    helpMat[a + u][b + v] = help

    return np.array(helpMat)


def idcTransformation(matrix):
    nmatrix = matrix.copy()
    helpMat = np.zeros((8, 8))

    for a in range(0, nmatrix.shape[0], 8):
        for b in range(0, nmatrix.shape[1], 8):
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

                            help += cu * cv * nmatrix[a + u][b + v] * \
                                math.cos(((2 * x + 1) * u * math.pi) / 16) * \
                                math.cos(((2 * y + 1) * v * math. pi) / 16)

                    helpMat[x][y] = help * 1 / 4

            for x in range(0, 8):
                for y in range(0, 8):
                    nmatrix[a + x][b + y] = helpMat[x][y]

    return nmatrix


def quant(matrix, chrominanz):

    help = np.zeros((matrix.shape[0], matrix.shape[1]))

    if chrominanz:
        qm = np.array([[16, 11, 10, 16, 24, 40, 51,
                        61], [12, 12, 14, 19, 26, 48, 60,
                              55], [14, 13, 16, 24, 40, 57, 69,
                                    56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103,
                        77], [24, 35, 55, 64, 81, 104, 113,
                              92], [49, 64, 78, 87, 103, 121, 120,
                                    101], [72, 92, 95, 98, 112, 100, 103, 99]])
    else:
        qm = np.array([[17, 18, 24, 47, 99, 99, 99,
                        99], [18, 21, 26, 66, 99, 99, 99,
                              99], [24, 26, 56, 99, 99, 99, 99,
                                    99], [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99,
                        99], [99, 99, 99, 99, 99, 99, 99,
                              99], [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

    for x in range(0, matrix.shape[0], 8):
        for y in range(0, matrix.shape[1], 8):
            for u in range(0, 8):
                for v in range(0, 8):
                    help[x + u][y + v] = int(
                        round(matrix[x + u][y + v] / qm[u][v]))

    return np.array(help)


def dequant(matrix, chrominanz):
    nmatrix = matrix.copy()

    if chrominanz:
        qm = np.array([[16, 11, 10, 16, 24, 40, 51,
                        61], [12, 12, 14, 19, 26, 48, 60,
                              55], [14, 13, 16, 24, 40, 57, 69,
                                    56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103,
                        77], [24, 35, 55, 64, 81, 104, 113,
                              92], [49, 64, 78, 87, 103, 121, 120,
                                    101], [72, 92, 95, 98, 112, 100, 103, 99]])
    else:
        qm = np.array([[17, 18, 24, 47, 99, 99, 99,
                        99], [18, 21, 26, 66, 99, 99, 99,
                              99], [24, 26, 56, 99, 99, 99, 99,
                                    99], [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99,
                        99], [99, 99, 99, 99, 99, 99, 99,
                              99], [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

    for a in range(0, nmatrix.shape[0], 8):
        for b in range(0, nmatrix.shape[1], 8):
            for x in range(0, 8):
                for y in range(0, 8):
                    nmatrix[a + x][b + y] = nmatrix[a + x][b + y] * qm[x][y]
    return nmatrix


def codierung(matrix):
    return matrix.copy


def decodierung(matrix, table):
    return matrix.copy


def entropie(matrix):

    entropie = 0
    counts = Counter()
    flache_matrix = np.reshape(matrix, -1)

    for x in range(flache_matrix.shape[0]):
        counts[x] += 1

    probs = [float(c) / flache_matrix.shape[0] for c in counts.values()]

    for p in probs:
        if p > 0.:
            entropie -= p * math.log(p, 2)

    return entropie


def entscheidungsgehalt(matrix):

    counts = Counter()
    flache_matrix = np.reshape(matrix, -1)

    for x in flache_matrix:
        counts[x] += 1

    lan = len(counts.keys())

    infogehalt = math.log(lan, 2)

    return infogehalt


def quellenredundanz(matrix):

    return (entropie(matrix) - entscheidungsgehalt(matrix))


def mse(Y, YH):
    return np.square(Y - YH).mean()


def psnr(Y, YH):
    max_val = 255
    fehler = mse(Y, YH)
    return 20 * math.log(max_val, 10) - 10 * math.log(fehler, 10)


def groesse(o):
    return sys.getsizeof(o)


def compFak(a, b):

    return (b / a)


def save(matrix, path):

    pic = Image.fromarray(np.uint8(matrix))
    pic.save(path)


def speichere_histogramm(label, hist):
    bildpfad = pfad(label + '.png')
    fig.savefig(bildpfad)
    return bildpfad


def pfad(datei):

    return os.path.abspath(ordner + '/' + datei)


def drucke_bild(label, array):

    img = Image.fromarray(array.astype('uint8'))
    bildpfad = pfad(label + '.png')
    img.save(bildpfad)
    return bildpfad


def histogramm(array):

    plt.close('all')
    davi.set_style('whitegrid')

    eindim = np.reshape(array, -1)

    plot = davi.distplot(eindim)
    fig = plot.get_figure()

    return fig
