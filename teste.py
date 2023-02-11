''' Duarte Emanuel Ramos Meneses - 2019216949
    Inês Martins Marçal - 2019215917
    Patricia Beatriz Silva Costa - 2019213995
    Multimédia - TP1 - 2021/2022              '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.fftpack as fft
import math

test = False

mat = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]

q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
       [12, 12, 14, 19, 26, 58, 60, 55],
       [14, 13, 16, 24, 40, 57, 69, 56],
       [14, 17, 22, 29, 51, 87, 80, 62],
       [18, 22, 37, 56, 68, 109, 103, 77],
       [24, 35, 55, 64, 81, 104, 113, 92],
       [49, 64, 78, 87, 103, 121, 120, 101],
       [72, 92, 95, 98, 112, 100, 103, 99]])

q_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
          [18, 21, 26, 66, 99, 99, 99, 99],
          [24, 26, 56, 99, 99, 99, 99, 99],
          [47, 66, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99]])


def menu_inicial():
    option = 0
    while (option < 1 or option > 3):
        print("Que imagem deseja ler?\n1- barn_mountains.bmp\n2- logo.bmp\n3- peppers.bmp\n")
        option=input("Opção: ")
        if option.isnumeric():
            option = int(option)
            if (option < 1 or option > 3):
                print("Opção inválida. Escolha outra opção.\n")
                option = 0
        else: 
            print("Opção inválida. Escolha outra opção.\n")
            option = 0  
    img_name = ''
    if (option == 1):
        img_name = 'barn_mountains.bmp'
    elif (option == 2):
        img_name = 'logo.bmp'
    else:
        img_name = 'peppers.bmp'

    return img_name


def show_img(title, img):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()


def show_colormap(img, colormap, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img, colormap)
    plt.show()


def colormap(r, g, b, name):
    return clr.LinearSegmentedColormap.from_list(name, [(0,0,0), (r,g,b)], 256)


def separate_rgb(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def join_rgb(R, G, B, type):
    joined = np.zeros([R.shape[0], R.shape[1], 3], dtype=type)
    joined[:, :, 0] = R[:, :]
    joined[:, :, 1] = G[:, :]
    joined[:, :, 2] = B[:, :]

    return joined


def padding(R, G, B):  
    lines = R.shape[0]
    final_lines = 0
    if (lines % 16 != 0):
        final_lines = lines + (16 - lines % 16)
    else:
        final_lines = lines

    columns = R.shape[1]
    final_columns = 0
    if (columns % 16 != 0):
        final_columns = columns + (16 - columns % 16)
    else:
        final_columns = columns
    
    img_padd = np.zeros([final_lines, final_columns, 3], dtype=R.dtype)

    img_padd[:R.shape[0], :R.shape[1], 0] = R[:, :]  #R
    img_padd[:G.shape[0], :G.shape[1], 1] = G[:, :]  #G
    img_padd[:B.shape[0], :B.shape[1], 2] = B[:, :]  #B

    a = R.shape[0]
    b = R.shape[1]

    r_plus = np.repeat([R[-1, :]], (final_lines-a), axis=0)
    R = np.vstack((R, r_plus))

    g_plus = np.repeat([G[-1, :]], (final_lines-a), axis=0)
    G = np.vstack((G, g_plus))

    b_plus = np.repeat([B[-1, :]], (final_lines-a), axis=0)
    B = np.vstack((B, b_plus))

    r_plus = np.repeat(R[:, -1:], (final_columns-b), axis=1)
    R = np.hstack((R, r_plus))

    g_plus = np.repeat(G[:, -1:], (final_columns-b), axis=1)
    G = np.hstack((G, g_plus))

    b_plus = np.repeat(B[:, -1:], (final_columns-b), axis=1)
    B = np.hstack((B, b_plus))


    if test:
        img_padd = join_rgb(R, G, B, R.dtype)
        show_img("Padding", list(img_padd))
        print("Dimensão da imagem após o padding: " + str(img_padd.shape))

    return R, G, B


def reverse_padding(R, G, B, lines, columns):
    R_unp = R[:lines, :columns]
    G_unp = G[:lines, :columns]
    B_unp = B[:lines, :columns]

    if test:
        print("Dimensão da imagem após a reversão do padding: " + str(R_unp.shape))

    return R_unp, G_unp, B_unp


def YCbCr(R_p, G_p, B_p):

    if test:
        print("Pixel [0][0] antes da conversão de RGB para YCbCr\n\tValor original do R: "+str(R_p[0][0])+"\n\tValor original do G: "+str(G_p[0][0])+"\n\tValor original do B: "+str(B_p[0][0])+'\n')

    Y = mat[0][0]*R_p + mat[0][1]*G_p + mat[0][2]*B_p 
    Cb = mat[1][0]*R_p + mat[1][1]*G_p + mat[1][2]*B_p + 128
    Cr = mat[2][0]*R_p + mat[2][1]*G_p + mat[2][2]*B_p + 128

    if test:
        cmGray = colormap(1, 1, 1, 'myGray')
        show_colormap(Y, cmGray, "Canal Y da imagem com colormap cinzento")
        show_colormap(Cb, cmGray, "Canal Cb da imagem com colormap cinzento")
        show_colormap(Cr, cmGray, "Canal Cr da imagem com colormap cinzento")
    print(Y[0][0], Cb[0][0], Cr[0][0])
    return Y, Cb, Cr


def reverse_ycbcr(Y, Cb, Cr):
    mat_i = np.linalg.inv(mat)

    R = (mat_i[0][0] * Y) + (mat_i[0][1] * (Cb - 128)) + (mat_i[0][2] * (Cr - 128))
    G = (mat_i[1][0] * Y) + (mat_i[1][1] * (Cb - 128)) + (mat_i[1][2] * (Cr - 128))
    B = (mat_i[2][0] * Y) + (mat_i[2][1] * (Cb - 128)) + (mat_i[2][2] * (Cr - 128))

    R = R.round()
    R[R>255] = 255
    R[R<0] = 0
    R = R.astype(np.uint8)

    G = G.round()
    G[G>255] = 255
    G[G<0] = 0
    G = G.astype(np.uint8)

    B = B.round()
    B[B>255] = 255
    B[B<0] = 0
    B = B.astype(np.uint8)

    return R, G, B


def encoder(img_name):
    plt.close('all')
    
    img = plt.imread(img_name)

    show_img("Imagem Original: " + img_name, img)

    if test:
        print("Dimensão original da imagem: " + str(img.shape))

    #Colormaps do RGB
    cmRed = colormap(1, 0, 0, 'myRed')
    cmGreen = colormap(0, 1, 0, 'myGreen')
    cmBlue = colormap(0, 0, 1, 'myBlue')

    if test:
        #Ver a imagem original com um colormap escolhido
        show_colormap(img, cmRed, "Imagem original com colormap " + "vermelho")

    #Separação dos canais RGB
    R, G, B = separate_rgb(img)

    if test:
        #Ver cada um dos canais com o colormap adequado
        show_colormap(R, cmRed, "Canal R da imagem original com colormap vermelho")
        show_colormap(G, cmGreen, "Canal G da imagem original com colormap verde")
        show_colormap(B, cmBlue, "Canal B da imagem original com colormap azul")

    #Padding da imagem
    R_p, G_p, B_p = padding(R, G, B)
    
    #Conversão para modelo YCbCr
    Y, Cb, Cr = YCbCr(R_p, G_p, B_p)

    
    return img, Y, Cb, Cr, img.shape[0], img.shape[1]


def decoder(Y, Cb, Cr, lines, columns):

    R, G, B = reverse_ycbcr(Y, Cb, Cr)
    if test:
        img_un_ycbcr = join_rgb(R, G, B, R.dtype)
        show_img("Conversão de YCbCr para RGB", list(img_un_ycbcr))

    R_unp, G_unp, B_unp = reverse_padding(R, G, B, lines, columns)
    if test:
        img_unp = join_rgb(R_unp, G_unp, B_unp, R_unp.dtype)
        show_img("Reversão do padding", img_unp)

    #Junção dos canais RGB de forma a obter a imagem original
    joined = join_rgb(R_unp, G_unp, B_unp, R_unp.dtype)
    show_img("Imagem descomprimida", list(joined))

    return Y_u, joined



def main():
    img_name = menu_inicial()
    img, Y, Cb, Cr, lines, columns= encoder(img_name)
    Y_dec, decoded = decoder(Y, Cb, Cr, lines, columns)




if __name__ == "__main__":
    main()