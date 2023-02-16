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

    ycbcr=mostraYCbCr(Y,Cb,Cr)

    grayCm = colorMap('gray', [(0,0,0),(1,1,1)], 256)
    
    show(Y,"canal y no colormap cinza",grayCm)
    show(Cb,"canal cb no colormap cinza",grayCm)
    show(Cr,"canal cr no colormap cinza",grayCm)

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
        repLine = np.repeat(lastLine, fillLines, axis=0) # repete a ultima linha x vezes
        pimage = np.vstack((pimage, repLine)) #junta / concateneação
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
    g = np.round(g).astype(np.uint8) #converte para int


    b = (matrixInv[2][0] * Y) + (matrixInv[2][1] * (Cb - 128)) + (matrixInv[2][2] * (Cr - 128))
    b[b > 255] = 255
    b[b < 0] = 0
    b = np.round(b).astype(np.uint8)

    print("Pixel [0][0] depois: R:"+str(r[0][0])+" G:"+str(g[0][0])+" B:"+str(b[0][0]))

    return r, g, b

def main():
    print("Escolha a imagem que deseja ler?\n1- logo.bmp\n2- barn_mountains.bmp\n3- peppers.bmp\n")
    opt=input("Opção: ")
    img = ""
    if(opt.isnumeric()):
        if(int(opt)>0 and int(opt)<4):
            opt = int(opt)
            if(opt == 1): img = "logo.bmp"
            elif(opt==2):  img = "barn_mountains.bmp"
            elif(opt==3):  img = "peppers.bmp"
        else:
            print("Opcão inválida\n")
            return 1
    else:
        print("Opcão inválida\n")
        return 1
        
    y, cb, cr, line, col=encoder(img)
    
    decoder(line, col, y, cb, cr)

if __name__=="__main__":
    main()