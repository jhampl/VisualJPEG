# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import math
from collections import Counter
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
from itertools import groupby
import json


# Encoder
def komprimiere(datei, *op_zielpfad):
    # Liste zur Aufnahme aller Schritte
    schritte = []
    # Einlesen des Bilds
    bild = np.array(Image.open(datei))

    # Reflektions-Padding des Bilds fuer die 8x8 Blockbildung
    bild_pad = pad(bild)
    schritte.extend([bild_pad, bild_pad[:,:,0],  bild_pad[:,:,1], bild_pad[:,:,2]])

    # Transformation in YCbCr
    ycbcr = frHinTransformation(bild_pad)
    schritte.extend([ycbcr, ycbcr[:,:,0],  ycbcr[:,:,1], ycbcr[:,:,2]])

    # Paralleles Verarbeiten der Bildkomponenten,
    # siehe Methode "_komprimiereKomponente"
    pool = Pool(3)
    # Markieren der Komponenten
    komponenten = [[ycbcr[:,:,0], 'Y'], [ycbcr[:,:,1], 'Cb'], [ycbcr[:,:,2], 'Cr']]
    ergebnisse = pool.map(_komprimiereKomponente, komponenten)
    # Sammeln der Unterschritte
    ycbcr_sub = [ergebnisse[0][1][0], ergebnisse[1][1][0], ergebnisse[2][1][0]]
    dct  = [ergebnisse[0][1][1], ergebnisse[1][1][1], ergebnisse[2][1][1]]
    dct_quant  = [ergebnisse[0][1][2], ergebnisse[1][1][2], ergebnisse[2][1][2]]
    schritte.extend(ycbcr_sub)
    schritte.extend(dct)
    schritte.extend(dct_quant)
    
    codierung = [ergebnisse[0][0], ergebnisse[1][0], ergebnisse[2][0]]

    # Speichern der Komprimierung
    # Standard Pfad, wenn keiner angegeben
    if len(op_zielpfad) == 0:
        bildname = datei.split('/')[len(datei.split('/'))-1].split('.')[0]
        zielpfad = os.path.abspath('./Ergebnisse/' + bildname + '_codiert.json')
    else:
        zielpfad = op_zielpfad[0]
    # Schreiben der codierten Daten in eine Datei
    with open(zielpfad, 'w') as jpeg:
        json.dump(codierung, jpeg)

    return zielpfad, schritte

# Komponenten-Encoder
def _komprimiereKomponente(komponente):
    schritte = []
    ycbcr = komponente[0].copy()
    chrominanz = komponente[1] != 'Y'

    # Test ob Chrominanz-Komponente
    if chrominanz:
        # 4:2:0 Unterabtasten der Chrominanz
        ycbcr_sub = unterabtastung(ycbcr)
    else:
        ycbcr_sub = ycbcr
    schritte.append(ycbcr_sub)
    
    # Diskrete Cosinus Transformation in 8x8 Bloecken
    dct = dcTransformation(ycbcr_sub)
    schritte.append(dct)

    # Quantisierung der Koeffizienten
    dct_quant = quantisierung(dct, chrominanz)
    schritte.append(dct_quant)

    # Codierung der quantisierten Koeffizienten
    dct_cod, groesse = codierung(dct_quant)
    schritte.append(dct_cod)


    return [[dct_cod, groesse], schritte]


# Decoder
def dekomprimiere(datei, *op_zielpfad):
    # Einlesen der JPEG Datei
    with open(datei, 'r') as jpeg:
        codierung = json.load(jpeg)

    # Liste zur Aufnahme aller Schritte
    schritte = []

    # Paralleles Verarbeiten der Bildkomponenten,
    # siehe Methode "_dekomprimiereKomponente"
    pool = Pool(3)
    # Markieren der Komponenten
    cod_komponenten = [[codierung[0], 'Y'], [codierung[1], 'Cb'], [codierung[2], 'Cr']]
    ergebnisse= pool.map(_dekomprimiereKomponente, cod_komponenten)

    # Sammeln der Unterschritte
    dct_dequant = [ergebnisse[0][1][0], ergebnisse[1][1][0], ergebnisse[2][1][0]]
    idct  = [ergebnisse[0][1][1], ergebnisse[1][1][1], ergebnisse[2][1][1]]
    ycbcr_up = [ergebnisse[0][1][2], ergebnisse[1][1][2], ergebnisse[2][1][2]]
    schritte.extend(dct_dequant)
    schritte.extend(idct)
    schritte.extend(ycbcr_up)

    ycbcr = [ergebnisse[0][0], ergebnisse[1][0], ergebnisse[2][0]]

    # Ruecktransformation in RGB
    bild = frRueckTransformation(ycbcr)
    schritte.extend([bild, bild[:,:,0],  bild[:,:,1], bild[:,:,2]])

    # Speichern als PNG
    # Standard zielpfad, wenn keiner angegeben
    if len(op_zielpfad) == 0:
        bildname = datei.split('/')[len(datei.split('/'))-1].split('.')[0]
        zielpfad = os.path.abspath('./Ergebnisse/' + bildname + '_komprimiert.png')
    else:
        zielpfad = op_zielpfad[0]
    img = Image.fromarray(bild)
    img.save(zielpfad)

    return zielpfad, schritte


# Komponenten-Decoder
def _dekomprimiereKomponente(komponente):
    schritte = []
    dct_cod = komponente[0][0].copy()
    groesse = komponente[0][1]
    chrominanz = komponente[1] != 'Y'

    # Decodierung der Koeffizienten
    dct_quant = decodierung(dct_cod, groesse)

    # Dequantisierung der Koeffizienten
    dct_dequant = dequantisierung(dct_quant, chrominanz)
    schritte.append(dct_dequant)

    # Inverse diskrete Cosinus Transformation in 8x8 Bloecken
    idct = idcTransformation(dct_dequant)
    schritte.append(idct)

    # Test ob Chrominanz-Komponent
    if chrominanz:
        # Ueberabtastung der Chrominanz/Interpolation
        ycbcr_up = ueberabtastung(idct)
    else:
        ycbcr_up = idct
    schritte.append(ycbcr_up)

    return [ycbcr_up, schritte]


def frHinTransformation(matrix):
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


def frRueckTransformation(komponente):
    matrix = np.zeros((komponente[0].shape[0],  komponente[0].shape[1], 3)).astype(np.float64) 
    matrix[:,:,0] = komponente[0]
    matrix[:,:,1] = komponente[1]
    matrix[:,:,2] = komponente[2]
    ycbcr = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.722, 0]])
    matrix[:, :, [1, 2]] -= 128

    for a in range(matrix.shape[0]):
        for b in range(matrix.shape[1]):
            np.matmul(ycbcr, np.array([matrix[a][b][0], matrix[a][b][1], matrix[a][b][2]]), matrix[a][b])

    np.putmask(matrix, matrix > 255, 255)
    np.putmask(matrix, matrix < 0, 0)

    return matrix.astype(np.uint8)


def unterabtastung(matrix):
    return matrix[::2, ::2]


def ueberabtastung(matrix):
    return matrix.repeat(2, axis=0).repeat(2, axis=1)


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


def quantisierung(matrix, chrominanz):
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


def dequantisierung(matrix, chrominanz):
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
    table = [[0,1,0,0,1,2,3,2,1,0,0,1,2,3,4,5,4,3,2,1,0,0,1,2,3,4,5,6,7,6,5,
        4,3,2,1,0,1,2,3,4,5,6,7,7,6,5,4,3,2,3,4,5,6,7,7,6,5,4,5,6,7,7,6,7],
        [0,0,1,2,1,0,0,1,2,3,4,3,2,1,0,0,1,2,3,4,5,6,5,4,3,2,1,0,0,1,2,3,4,
        5,6,7,7,6,5,4,3,2,1,2,3,4,5,6,7,7,6,5,4,3,4,5,6,7,7,6,5,6,7,7]]
    array = []
    res = []

    for a in range(0, matrix.shape[0], 8):
        for b in range(0, matrix.shape[1], 8):
            for x in range(np.array(table).shape[1]):
                array.append(matrix[a + table[1][x]][b + table[0][x]])

    for k, i in groupby(array):
        run = list(i)
        if len(run) > 3:
            res.append("({},{})".format(len(run), k))
        else:
            res.extend(run)

    return res, [matrix.shape[0], matrix.shape[1]]


def decodierung(matrix, groesse):
    table = [[0,1,0,0,1,2,3,2,1,0,0,1,2,3,4,5,4,3,2,1,0,0,1,2,3,4,5,6,7,6,5,
        4,3,2,1,0,1,2,3,4,5,6,7,7,6,5,4,3,2,3,4,5,6,7,7,6,5,4,5,6,7,7,6,7],
        [0,0,1,2,1,0,0,1,2,3,4,3,2,1,0,0,1,2,3,4,5,6,5,4,3,2,1,0,0,1,2,3,4,
        5,6,7,7,6,5,4,3,2,1,2,3,4,5,6,7,7,6,5,4,3,4,5,6,7,7,6,5,6,7,7]]
    alt = []
    nmatrix = np.zeros((groesse[0], groesse[1]))
    
    for x in matrix:
        if type(x) is str:
            val = x.split(',')
            firstVal = str(val[0])[1:]
            secondVal = str(val[1])[0:len(str(val[1])) - 1]
            for number in range(int(float(firstVal))):
                alt.append(float(secondVal))
        else:
            alt.append(x)

    i = 0
    for a in range(0, nmatrix.shape[0], 8):
        for b in range(0, nmatrix.shape[1], 8):
            for x in range(np.array(table).shape[1]):
                nmatrix[a + table[1][x]][b + table[0][x]] = float(alt[x+i])
            i += 64

    return np.array(nmatrix)


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


def compFak(a, b):
    return (b / a)
