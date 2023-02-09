import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as clr


matrix = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]


def encoder(source):
    return 0


def decoder():
    return 0


def readImage(source):
    image = img.imread(source)
    return image


def colorMap(name,colours,value):
    cm = clr.LinearSegmentedColormap.from_list(name, colours, value)
    return cm


def showImage(img,cMap = None):
    plt.figure()
    plt.imshow(img,cMap)
    plt.axis('off')
    plt.show()


def rgb_comp(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return r, g, b


def rgb_recons(r, g, b):
    [nl, nc] = r.shape
    imgRec = np.zeros((nl, nc, 3))
    imgRec[:, :, 0] = r
    imgRec[:, :, 1] = g
    imgRec[:, :, 2] = b
    return imgRec


def padding(comp):
    [nLine, nColumns, x] = comp.shape
    pComp = np.zeros([nLine, nColumns, x])

    fillLines = 32 - nLine % 32
    fillColumns = 32 - nColumns % 32

    if fillLines != 32 and fillColumns != 32:
        lastLine = comp[nLine - 1, :][np.newaxis, :]
        repLine = lastLine.repeat(fillLines, axis=0)
        pComp = np.vstack([comp, repLine])
        lastColumn = comp[nLine-1, :][:, np.newaxis]
        repColumn = lastColumn.repeat(fillColumns, axis=1)
        pComp = np.vstack([pComp,repColumn])
        return pComp

    elif fillLines != 32:
        lastLine = comp[nLine - 1, :][np.newaxis, :]
        repLine = lastLine.repeat(fillLines, axis=0)
        pComp = np.vstack([comp, repLine])

    elif fillColumns != 32:
        lastColumn = comp[nLine - 1, :][:, np.newaxis]
        repColumn = lastColumn.repeat(fillColumns, axis=1)
        pComp = np.vstack([comp, repColumn])

    return pComp




def rgbToYCbCr(r, g, b):
    Y = matrix[0][0]*r+matrix[0][0]*g+matrix[0][0]*b
    Cb = matrix[1][0]*r+matrix[1][0]*g+matrix[1][0]*b+128
    Cr = matrix[1][0]*r+matrix[1][0]*g+matrix[1][0]*b+128
    return Y, Cb, Cr


def YCbCrTorgb(Y, Cb, Cr):
    matrixInv = np.linalg.inv(matrix)

    r = matrixInv[0,0]*Y + matrixInv[0,1] * (Cb - 128) + matrixInv[0,2] * (Cr - 128)
    r[r > 255] = 255
    r[r < 0] = 0
    r = np.round(r).astype(np.uint8)

    g = matrixInv[1, 0] * Y + matrixInv[1, 1] * (Cb - 128) + matrixInv[1, 2] * (Cr - 128)
    g[g > 255] = 255
    g[g < 0] = 0
    g = np.round(g).astype(np.uint8)

    b = matrixInv[2, 0] * Y + matrixInv[2, 1] * (Cb - 128) + matrixInv[2, 2] * (Cr - 128)
    b[b > 255] = 255
    b[b < 0] = 0
    b = np.round(b).astype(np.uint8)
    return r, g, b
