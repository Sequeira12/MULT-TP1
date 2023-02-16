"""
Bruno Sequeira: 
Rui Santos: 2020225542
Tomás Dias:

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as clr
import math
import cv2


matrix = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]


def encoder(bmp):
    image = readImage(bmp)
    line = image.shape[0]
    col = image.shape[1]

    #colormap
    redCm = colorMap('red',[(0,0,0),(1,0,0)] ,256)
    greenCm = colorMap('green',[(0,0,0),(0,1,0)] ,256)
    blueCm = colorMap('blue',[(0,0,0),(0,0,1)] ,256)
    
    # 3.4 e 3.5: separacao das componentes RGB e visualizacao com o seu colorMap
    r, g, b = rgbComp(image)
    show(r,"Canal R com colorMap vermelho", redCm)
    show(g,"Canal G com colorMap verde", greenCm)
    show(b,"Canal B com colorMap azul", blueCm)
    
    pimage = padding(image)
    
    r, g, b = rgbComp(pimage)

    Y, Cb, Cr = rgbToYCbCr(r,g,b)

    #mostraYCbCr(Y,Cb,Cr)

    grayCm = colorMap('gray', [(0,0,0),(1,1,1)], 256)
    
    show(Y,"canal y no colormap cinza",grayCm)
    show(Cb,"canal cb no colormap cinza",grayCm)
    show(Cr,"canal cr no colormap cinza",grayCm)

    downsampling(Y,Cb,Cr,4,2,0, grayCm)

    return Y, Cb, Cr, line, col


def decoder(line, col,y, cb, cr):
    r,g,b = YCbCrTorgb(y,cb,cr)
    
    #reconstroi imagem com padding
    pimage = rgbRecons(r,g,b)

    #tira o padding
    image = inversePadding(pimage,line, col)

    show(image, "imagem descomprimida")
    return 0


# 3.1
def readImage(bmp):
    image = img.imread(bmp)
    return image


# 3.2
def colorMap(name, colours, value):
    cm = clr.LinearSegmentedColormap.from_list(name, colours, value)
    return cm

#3.3
def show(image, title, colormap=None):
    plt.figure()
    plt.title(title)
    plt.imshow(image,colormap)
    plt.axis('off')
    plt.show()


# 3.4
def rgbComp(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    return r, g, b


# 3.4
def rgbRecons(r, g, b):
    [nl, nc] = r.shape
    imgRec = np.zeros([nl, nc, 3], r.dtype)
    imgRec[:, :, 0] = r[:,:]
    imgRec[:, :, 1] = g[:,:]
    imgRec[:, :, 2] = b[:,:]
    return imgRec


# 4.1
def padding(image):
    [nLine, nColumns, x] = image.shape

    fillLines = 32 - nLine % 32
    fillColumns = 32 - nColumns % 32

    pimage = image
    if fillLines != 32:
        lastLine = pimage[nLine - 1, :][np.newaxis, :]
        repLine = np.repeat(lastLine, fillLines, axis=0)
        pimage = np.vstack((pimage, repLine))
    if fillColumns != 32:
        lastColumn = pimage[:, nColumns - 1][:, np.newaxis]
        repColumn = np.repeat(lastColumn, fillColumns, axis=1)
        pimage = np.hstack((pimage, repColumn))
    return pimage


# 4.1
def inversePadding(pimage,nLines,nColumns):
    image = pimage[:nLines, :nColumns]
    return image


# 5.1
def rgbToYCbCr(r, g, b):
    print("Pixel [0][0] antes: R:"+str(r[0][0])+" G:"+str(g[0][0])+" B:"+str(b[0][0]))
    Y = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b
    Cb = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b + 128
    Cr = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b + 128    
    return Y, Cb, Cr



def mostraYCbCr(Y,Cb,Cr):
    ycbcr = np.zeros((Y.shape[0], Y.shape[1], 3), dtype=np.uint8)
    ycbcr[:,:,0] = Y
    ycbcr[:,:,1] = Cb
    ycbcr[:,:,2] = Cr
    show(ycbcr,"Imagem em YCbCr")
    return ycbcr

# 5.1
def YCbCrTorgb(Y, Cb, Cr):
    matrixInv = np.linalg.inv(matrix)

    r = (matrixInv[0][0] * Y) + (matrixInv[0][1] * (Cb - 128)) + (matrixInv[0][2] * (Cr - 128))
    r[r > 255] = 255
    r[r < 0] = 0
    r = np.round(r).astype(np.uint8)

    g = (matrixInv[1][0] * Y) + (matrixInv[1][1] * (Cb - 128)) + (matrixInv[1][2] * (Cr - 128))
    g[g > 255] = 255
    g[g < 0] = 0
    g = np.round(g).astype(np.uint8)

    b = (matrixInv[2][0] * Y) + (matrixInv[2][1] * (Cb - 128)) + (matrixInv[2][2] * (Cr - 128))
    b[b > 255] = 255
    b[b < 0] = 0
    b = np.round(b).astype(np.uint8)

    print("Pixel [0][0] depois: R:"+str(r[0][0])+" G:"+str(g[0][0])+" B:"+str(b[0][0]))

    return r, g, b



def downsampling(Y, Cb, Cr,fY,fCr,fCb,colormap):
    """
    Parameters:
    source: Input Image array (Single-channel, 8-bit or floating-point) 
    dsize: Size of the output array
    dest: Output array (Similar to the dimensions and type of Input image array) [optional]
    fx: Scale factor along the horizontal axis  [optional]
    fy: Scale factor along the vertical axis  [optional]
    interpolation: One of the above interpolation methods  [optional]
    """
    Y_d=Y
    fy=1
    if(fCb == 0): 
        fy=fCr/fY
        fCb=fCr

    x1=int(Cb.shape[1]*(fCb/fY))
    y2=int(Cb.shape[0]*fy)
    dim1=(x1,y2)

    x2=int(Cr.shape[1]*(fCr/fY))
    y2=int(Cr.shape[0]*fy)
    dim2=(x2,y2)

    Cb_d = cv2.resize(Cb,dim1,fx=fCb/fY, fy=fy, interpolation=cv2.INTER_LINEAR)
    Cr_d = cv2.resize(Cr,dim2,fx=fCr/fY, fy=fy, interpolation=cv2.INTER_LINEAR)
    

    print(fCb/fY, fCr/fY, fy)
    show(Cb_d,"subamostragem cb", colormap)
    show(Cr_d,"subamostragem cr", colormap)
    return Y_d, Cb_d, Cr_d

def upsampling():
    pass


def main():
    "logo.bmp" "barn_mountains.bmp""peppers.bmp"
    y, cb, cr, line, col=encoder("barn_mountains.bmp")
    
    decoder(line, col, y, cb, cr)

if __name__=="__main__":
    main()