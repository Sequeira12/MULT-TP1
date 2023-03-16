"""
Bruno Sequeira: 2020235721
Rui Santos: 2020225542
Tomás Dias: 2020215701

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as clr
import math
import scipy.fftpack as f
import cv2


fatorQ=100
nBlocos = 8

"peppers.bmp" "logo.bmp" "barn_mountains.bmp"
a_ver = "barn_mountains.bmp"

matrizQuantY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [49, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]])

matrizQuantCbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                            [18, 21, 26, 66, 99, 99, 99, 99],
                            [24, 26, 56, 99, 99, 99, 99, 99],
                            [47, 66, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99]])

matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])


def encoder(bmp):
    image = readImage(bmp)
    line = image.shape[0]
    col = image.shape[1]

    # colormap
    redCm = colorMap('red', [(0, 0, 0), (1, 0, 0)], 256)
    greenCm = colorMap('green', [(0, 0, 0), (0, 1, 0)], 256)
    blueCm = colorMap('blue', [(0, 0, 0), (0, 0, 1)], 256)

    # 3.4 e 3.5: separacao das componentes RGB e visualizacao com o seu colorMap
    r, g, b = rgbComp(image)
    show(r, "Canal R com colorMap vermelho", redCm)
    show(g, "Canal G com colorMap verde", greenCm)
    show(b, "Canal B com colorMap azul", blueCm)

    pimage = padding(image)

    r, g, b = rgbComp(pimage)

    Y, Cb, Cr = rgbToYCbCr(r, g, b)

    # mostraYCbCr(Y,Cb,Cr)

    grayCm = colorMap('gray', [(0, 0, 0), (1, 1, 1)], 256)

    show(Y, "canal y no colormap cinza", grayCm)
    show(Cb, "canal cb no colormap cinza", grayCm)
    show(Cr, "canal cr no colormap cinza", grayCm)

    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, 4, 2, 2, grayCm)

    # Escolher 0,8,64 para aplicar a imagem inteira, blocos 8x8 e blocos 64x64 respetivamente
    Y_dct, Cb_dct, Cr_dct = dctblocos(Y_d, Cb_d, Cr_d, nBlocos, grayCm)

    Y_quant, Cb_quant, Cr_quant, matrizQuantY2, matrizQuantCbCr2 = quantizacao(Y_dct, Cb_dct, Cr_dct, nBlocos, grayCm, fatorQ,
                                                                               True)
    Y_dpcm,Cb_dpcm,Cr_dpcm = codificao_dpcm(Y_quant, Cb_quant, Cr_quant,nBlocos,0,grayCm,True)

    return image , Y,line, col, Y_dpcm, Cb_dpcm, Cr_dpcm, matrizQuantY2, matrizQuantCbCr2


def decoder(line, col, Y_dpcm, Cb_dpcm, Cr_dpcm, matrizQuantY2, matrizQuantCbCr2):
    grayCm = colorMap('gray', [(0, 0, 0), (1, 1, 1)], 256)

    Y_quant,Cb_quant,Cr_quant = codificao_dpcm(Y_dpcm, Cb_dpcm, Cr_dpcm,nBlocos,1,grayCm,True)

    Y_dct, Cb_dct, Cr_dt = inversoquantizacao(Y_quant, Cb_quant, Cr_quant, matrizQuantY2, matrizQuantCbCr2, nBlocos, grayCm,
                                              fatorQ, True)
    # Escolher 0,8,64 para aplicar a imagem inteira, blocos 8x8 e blocos 64x64 respetivamente
    Y_d, Cb_d, Cr_d = inversodctblocos(Y_dct, Cb_dct, Cr_dt, nBlocos, grayCm)

    y, cb, cr = upsampling(Y_d, Cb_d, Cr_d, 4, 2, 2, grayCm)

    r, g, b = YCbCrTorgb(y, cb, cr)

    # reconstroi imagem com padding
    pimage = rgbRecons(r, g, b)

    # tira o padding
    image = inversePadding(pimage, line, col)

    show(image, "imagem descomprimida")
    return image, y, cb, cr


# 3.1
def readImage(bmp):
    image = img.imread(bmp)
    return image


# 3.2
def colorMap(name, colours, value):
    cm = clr.LinearSegmentedColormap.from_list(name, colours, value)
    return cm


# 3.3
def show(image, title, colormap=None):
    plt.figure()
    plt.title(title)
    plt.imshow(image, colormap)
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
    imgRec[:, :, 0] = r[:, :]
    imgRec[:, :, 1] = g[:, :]
    imgRec[:, :, 2] = b[:, :]
    return imgRec


# 4.1
def padding(image):
    [nLine, nColumns, x] = image.shape

    fillLines = 32 - nLine % 32
    fillColumns = 32 - nColumns % 32

    pimage = image
    if fillLines != 32:
        lastLine = pimage[nLine - 1, :][np.newaxis, :]
        repLine = np.repeat(lastLine, fillLines, axis=0)  # repete a ultima linha x vezes
        pimage = np.vstack((pimage, repLine))  # junta / concateneação
    if fillColumns != 32:
        lastColumn = pimage[:, nColumns - 1][:, np.newaxis]
        repColumn = np.repeat(lastColumn, fillColumns, axis=1)
        pimage = np.hstack((pimage, repColumn))
    return pimage


# 4.1
def inversePadding(pimage, nLines, nColumns):
    image = pimage[:nLines, :nColumns]
    return image


# 5.1
def rgbToYCbCr(r, g, b):
    # print("Pixel [0][0] antes: R:"+str(r[0][0])+" G:"+str(g[0][0])+" B:"+str(b[0][0]))
    Y = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b
    Cb = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b + 128
    Cr = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b + 128
    return Y, Cb, Cr


def mostraYCbCr(Y, Cb, Cr):
    ycbcr = np.zeros((Y.shape[0], Y.shape[1], 3), dtype=np.uint8)
    ycbcr[:, :, 0] = Y
    ycbcr[:, :, 1] = Cb
    ycbcr[:, :, 2] = Cr
    show(ycbcr, "Imagem em YCbCr")
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
    g = np.round(g).astype(np.uint8)  # converte para int

    b = (matrixInv[2][0] * Y) + (matrixInv[2][1] * (Cb - 128)) + (matrixInv[2][2] * (Cr - 128))
    b[b > 255] = 255
    b[b < 0] = 0
    b = np.round(b).astype(np.uint8)

    # print("Pixel [0][0] depois: R:"+str(r[0][0])+" G:"+str(g[0][0])+" B:"+str(b[0][0]))

    return r, g, b


def downsampling(Y, Cr, Cb, fY, fCr, fCb, colormap):
    """
    Parameters:
    source: Input Image array (Single-channel, 8-bit or floating-point)
    dsize: Size of the output array
    dest: Output array (Similar to the dimensions and type of Input image array) [optional]
    fx: Scale factor along the horizontal axis  [optional]
    fy: Scale factor along the vertical axis  [optional]
    interpolation: One of the above interpolation methods  [optional]
    """
    Y_d = Y
    fy = 1
    if (fCb == 0):
        fy = fCr / fY
        fCb = fCr

    x1 = int(Cb.shape[1] * (fCb / fY))
    y2 = int(Cb.shape[0] * fy)
    dim1 = (x1, y2)

    x2 = int(Cr.shape[1] * (fCr / fY))
    y2 = int(Cr.shape[0] * fy)
    dim2 = (x2, y2)

    Cb_d = cv2.resize(Cb, dim1, fx=fCb / fY, fy=fy, interpolation=cv2.INTER_LINEAR)
    Cr_d = cv2.resize(Cr, dim2, fx=fCr / fY, fy=fy, interpolation=cv2.INTER_LINEAR)

    show(Y_d, "subamostragem Y", colormap)
    show(Cb_d, "subamostragem cb", colormap)
    show(Cr_d, "subamostragem cr", colormap)

    return Y_d, Cb_d, Cr_d


def upsampling(Y_d, Cr_d, Cb_d, fY, fCr, fCb, colormap):
    Y = Y_d
    fy = 1
    if (fCb == 0):
        fy = fY / fCr
        fCb = fCr

    x1 = int(Cb_d.shape[1] * (fY / fCb))
    y2 = int(Cb_d.shape[0] * (fy))
    dim1 = (x1, y2)

    x2 = int(Cr_d.shape[1] * (fY / fCr))
    y2 = int(Cr_d.shape[0] * (fy))
    dim2 = (x2, y2)

    Cb = cv2.resize(Cb_d, dim1, fx=fY / fCb, fy=fy, interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_d, dim2, fx=fY / fCr, fy=fy, interpolation=cv2.INTER_LINEAR)

    #print(fy / fCb, fY / fCr, fy)
    #print(Cb.shape[0], Cb.shape[1], Cb_d.shape[0], Cb_d.shape[1])
    #print(Cr.shape[0], Cr.shape[1], Cr_d.shape[0], Cr_d.shape[1])

    show(Y, "reconstrucao Y", colormap)
    show(Cb, "reconstrucao cb", colormap)
    show(Cr, "reconstrucao cr", colormap)
    return Y, Cb, Cr


# 7.1

def dct(X):
    return f.dct(f.dct(X, norm="ortho").T, norm="ortho").T


def inversa_dct(X):
    return f.idct(f.idct(X, norm="ortho").T, norm="ortho").T


# funcao que percorre os canais e calcula a dct em bloco
def percorreDCTblocos(ch, blocos, dctORidct):
    ch_dct = np.zeros((ch.shape[0], ch.shape[1]))
    range1 = int(ch.shape[0] / blocos)
    range2 = int(ch.shape[1] / blocos)

    for i in range(range1):
        for j in range(range2):
            if (dctORidct == 1):
                ch_dct[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] = dct(
                    ch[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos])
            else:
                ch_dct[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] = inversa_dct(
                    ch[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos])

    return ch_dct


# 7.3
def dctblocos(Y_d, Cb_d, Cr_d, blocos, colormap):
    if (blocos == 0):
        Cb_dct = dct(Cb_d)
        Cr_dct = dct(Cr_d)
        Y_dct = dct(Y_d)
    else:
        Cb_dct = percorreDCTblocos(Cb_d, blocos, 1)
        Cr_dct = percorreDCTblocos(Cr_d, blocos, 1)
        Y_dct = percorreDCTblocos(Y_d, blocos, 1)

    # visualização as imagens usando uma transformação logarítmica
    logY = np.log(np.abs(Y_dct) + 0.0001)
    logCb = np.log(np.abs(Cb_dct) + 0.0001)
    logCr = np.log(np.abs(Cr_dct) + 0.0001)

    titulo = str(blocos) + "x" + str(blocos)
    show(logY, "Canal Y com DCT em blocos " + titulo, colormap)
    show(logCb, "Canal Cb com DCT em blocos " + titulo, colormap)
    show(logCr, "Canal Cr com DCT em blocos " + titulo, colormap)
    return Y_dct, Cb_dct, Cr_dct


def inversodctblocos(Y_dct, Cb_dct, Cr_dct, blocos, colormap):
    if (blocos == 0):
        Cb_d = inversa_dct(Cb_dct)
        Cr_d = inversa_dct(Cr_dct)
        Y_d = inversa_dct(Y_dct)
    else:
        Cb_d = percorreDCTblocos(Cb_dct, blocos, 0)
        Cr_d = percorreDCTblocos(Cr_dct, blocos, 0)
        Y_d = percorreDCTblocos(Y_dct, blocos, 0)

    # visualização as imagens usando uma transformação logarítmica
    logY = np.log(np.abs(Y_d) + 0.0001)
    logCb = np.log(np.abs(Cb_d) + 0.0001)
    logCr = np.log(np.abs(Cr_d) + 0.0001)

    titulo = str(blocos) + "x" + str(blocos)
    show(logY, "(inverso) Canal Y com DCT em blocos " + titulo, colormap)
    show(logCb, " (inverso) Canal Cb com DCT em blocos " + titulo, colormap)
    show(logCr, "(inverso)Canal Cr com DCT em blocos " + titulo, colormap)

    return Y_d, Cb_d, Cr_d


# 8.1  e 8.2
def percorreQuantblocos(ch, blocos, colormap, matriz, titulo, tipo, imprime):
    ch_Quant = np.zeros((ch.shape[0], ch.shape[1]))
    range1 = int(ch.shape[0] / blocos)
    range2 = int(ch.shape[1] / blocos)

    for i in range(range1):
        for j in range(range2):
            if tipo == 1:
                ch_Quant[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] = np.round(
                    (ch[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] / matriz)).astype(int)
            else:
                ch_Quant[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] = (ch[
                                                                                           i * blocos:i * blocos + blocos,
                                                                                           j * blocos:j * blocos + blocos] * matriz).astype(float)

    if (imprime):
        log = np.log(np.abs(ch_Quant) + 0.0001)
        show(log, titulo, colormap=colormap)
    return ch_Quant


def fatorQualidade(qf, matriz):
    if qf >= 50:
        sf = (100 - qf) / 50
    else:
        sf = 50 / qf

    if sf == 0:
        qs = np.ones((matriz.shape[0], matriz.shape[1]))
        qs = np.round(qs).astype(np.uint8)
        return qs
    else:
        qs = matriz * sf
        qs[qs > 255] = 255
        qs[qs < 1] = 1
        qs = np.round(qs).astype(np.uint8)
        return qs


def quantizacao(Y_dct, Cb_dct, Cr_dct, blocos, colormap, fator, imprime):
    matrizCbCr = fatorQualidade(fator, matrizQuantCbCr)
    matrizY = fatorQualidade(fator, matrizQuantY)

    titulo = "Canal Y Quantizacao " + str(blocos) + "x" + str(blocos) + "  - fator: " + str(fator)
    Y_quant = percorreQuantblocos(Y_dct, blocos, colormap, matrizY, titulo, 1, imprime)

    titulo = "Canal Cb Quantizacao " + str(blocos) + "x" + str(blocos) + "  - fator: " + str(fator)
    Cb_quant = percorreQuantblocos(Cb_dct, blocos, colormap, matrizCbCr, titulo, 1, imprime)

    titulo = "Canal Cr Quantizacao " + str(blocos) + "x" + str(blocos) + "  - fator: " + str(fator)
    Cr_quant = percorreQuantblocos(Cr_dct, blocos, colormap, matrizCbCr, titulo, 1, imprime)

    return Y_quant, Cb_quant, Cr_quant, matrizY, matrizCbCr


def inversoquantizacao(Y_quant, Cb_quant, Cr_quant, matrizY, matrizCbCr, blocos, colormap, fator, imprime):
    titulo = "Canal Y Quantizacao (INV) " + str(blocos) + "x" + str(blocos) + "  - fator: " + str(fator)
    Y_dct = percorreQuantblocos(Y_quant, blocos, colormap, matrizY, titulo, 2, imprime)

    titulo = "Canal Cb Quantizacao (INV)" + str(blocos) + "x" + str(blocos) + "  - fator: " + str(fator)
    Cb_dct = percorreQuantblocos(Cb_quant, blocos, colormap, matrizCbCr, titulo, 2, imprime)

    titulo = "Canal Cr Quantizacao (INV)" + str(blocos) + "x" + str(blocos) + "  - fator: " + str(fator)
    Cr_dct = percorreQuantblocos(Cr_quant, blocos, colormap, matrizCbCr, titulo, 2, imprime)

    return Y_dct, Cb_dct, Cr_dct


# 9
def encode_dpcmblocos(ch,blocos):
    ch_dpcm = np.copy(ch)
    for i in range(blocos):
        for j in range(blocos):
                if j!= 0:
                    ch_dpcm[i, j] = ch[i, j]-ch[i, j-1]
                else:
                    if i != 0:
                        ch_dpcm[i,j] = ch[i,j]-ch[i-1,-1]
    return ch_dpcm


def decode_dpcmblocos(ch_dpcm,blocos):
    ch = np.copy(ch_dpcm)
    for i in range(blocos):
        for j in range(blocos):
            if j != 0:
                ch[i, j] = ch_dpcm[i, j] + ch[i, j - 1]
            else:
                if i != 0:
                    ch[i, j] = ch_dpcm[i, j] + ch[i - 1, -1]
    return ch


def percorredpcmblocos(ch, blocos,codeOrdecode,titulo,colormap,imprime):
    ch_dpcm = np.zeros((ch.shape[0], ch.shape[1]))
    range1 = int(ch.shape[0] / blocos)
    range2 = int(ch.shape[1] / blocos)
    for i in range(range1):
        for j in range(range2):
            if codeOrdecode == 0:
                ch_dpcm[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] = encode_dpcmblocos(
                    ch[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos], blocos)
            else:
                ch_dpcm[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos] = decode_dpcmblocos(
                    ch[i * blocos:i * blocos + blocos, j * blocos:j * blocos + blocos], blocos)
    if (imprime):
        log = np.log(np.abs(ch_dpcm) + 0.0001)
        show(log, titulo, colormap=colormap)

    return ch_dpcm


def codificao_dpcm(Y_quant, Cb_quant, Cr_quant,blocos,codeOrdecode,colormap,imprime):
    if codeOrdecode == 0:
        titulo = "Canal Y codificacao dpcm " + str(blocos) + "x" + str(blocos)
        Y_dpcm = percorredpcmblocos(Y_quant, blocos, codeOrdecode,titulo,colormap,imprime)

        titulo = "Canal Cb codificacao dpcm " + str(blocos) + "x" + str(blocos)
        Cb_dpcm = percorredpcmblocos(Cb_quant, blocos, codeOrdecode,titulo,colormap,imprime)

        titulo = "Canal Cr codificacao dpcm" + str(blocos) + "x" + str(blocos)
        Cr_dpcm = percorredpcmblocos(Cr_quant, blocos, codeOrdecode,titulo,colormap,imprime)
    else:
        titulo = "Canal Y apos descodificacao dpcm " + str(blocos) + "x" + str(blocos)
        Y_dpcm = percorredpcmblocos(Y_quant, blocos, codeOrdecode, titulo, colormap, imprime)

        titulo = "Canal Cb apos descodificacao dpcm " + str(blocos) + "x" + str(blocos)
        Cb_dpcm = percorredpcmblocos(Cb_quant, blocos, codeOrdecode, titulo, colormap, imprime)

        titulo = "Canal Cr apos descodificacao dpcm" + str(blocos) + "x" + str(blocos)
        Cr_dpcm = percorredpcmblocos(Cr_quant, blocos, codeOrdecode, titulo, colormap, imprime)

    return Y_dpcm,Cb_dpcm, Cr_dpcm



#10
def calculaErro(imagem, imagemRec, yrec,yor ,line, col):
    
    erro = abs(yor - yrec)
    print("Valor máximo de erro " + str(np.max(erro)))
    print("Valor médio de erro " + str(np.mean(erro)))
    grayCm = colorMap('gray', [(0, 0, 0), (1, 1, 1)], 256)

    show(erro,"erro",grayCm)
    #print(type(imagem[0][0]), type(imagemRec[0][0]))
    mse = (np.sum((imagem - imagemRec)**2)) / (line*col)
    rmse = math.sqrt(mse)
    p = (np.sum((imagem)**2)) / (line*col)
    snr = 10 * math.log10(p/mse)
    psnr = 10 * math.log10((np.max(imagem)**2)/mse)
    
    
    print ("MSE:", mse)
    print ("RMSE:", rmse)
    print ("SNR:", snr)
    print ("PSNR:", psnr)



def main():
    image,Yor, line, col, Y_dpcm, Cb_dpcm, Cr_dpcm, matrizQuantY2, matrizQuantCbCr2 = encoder(a_ver)

    imagemRec, y_rec, cb, cr = decoder(line, col, Y_dpcm, Cb_dpcm, Cr_dpcm, matrizQuantY2, matrizQuantCbCr2)

    image_f= image.astype(np.float64)

    calculaErro(image_f,imagemRec, y_rec, Yor, line, col)




if __name__ == "__main__":
    main()
