#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

#imagens tem 384x288 pixels, 256 grayscale

###################################################
class ImagemIntegral:
    def __init__(self, img):
        height, width = img.shape #we have H rows of W columns
        self.II = np.zeros( (height, width) )
        self.II2 = np.zeros( (height, width) )
        self.II[0,0] = img[0,0]
        self.II2[0,0] = img[0,0]**2
        for i in xrange( 1, height ):
            self.II[i,0] = self.II[i-1,0] + img[i,0]
            self.II2[i,0] = self.II2[i-1,0] + img[i,0]**2
        for i in xrange( 1, width ):
            self.II[0,i] = self.II[0,i-1] + img[0,i]
            self.II2[0,i] = self.II2[0,i-1] + img[0,i]**2
            
        for i in xrange(1, height):
            for j in xrange(1, width):
                self.II[i,j] = self.II[i,j-1] + self.II[i-1,j] + img[i,j] - self.II[i-1,j-1]
                self.II2[i,j] = self.II2[i,j-1] + self.II2[i-1,j] + (img[i,j]**2) - self.II2[i-1,j-1]
                
    def __getitem__(self, ij):
        i, j = ij
        if i < 0 or j < 0:
            return 0
        if i >= self.II.shape[0]:
            i = -1
        if j >= self.II.shape[1]:
            j = -1
        try:
            return self.II[i,j]
        except:
            return 0.0
            
    def getVar(self, i, j):
        if i < 0 or j < 0:
            return 0
        if i >= self.II2.shape[0]:
            i = -1
        if j >= self.II2.shape[1]:
            j = -1
        try:
            return self.II2[i,j]
        except:
            return 0

    def area(self, i, j, w, h):
        return self[i+w-1, j+h-1] - self[i+w-1, j-1] - self[i-1, j+h-1] + self[i-1, j-1]
        
    def media(self, i, j, w, h):
        return self.area(i,j,w,h) / (w*h)
        
    def desvioPadrao(self, i, j, w, h):
        n = w * h
        var = self.getVar(i+w-1, j+h-1) - self.getVar(i+w-1, j-1) - self.getVar(i-1, j+h-1) + self.getVar(i-1, j-1) #self.getVar(i,j)
        smean = self.media(i,j,w,h)**2
        #print "n=%s :: var=%s :: smean=%s" % (n,var,smean)
        stddev = np.sqrt( (1.0/n)*(var - (n*smean ) ) )
        if str(stddev) == 'nan':
            return 0.0
        return stddev

###################################################
def enhancementCoefficient(std):
    if 0 <= std < 20:
        return 3.0 / ( (0.005*((std-20)**2)) + 1 )
    elif 20 <= std < 60:
        return 3.0 / ( (0.00125*((std-20)**2)) + 1 )
    elif std >= 60:
        return 1.0
    raise Exception("Enhancement Coefficient not defined for std="+str(std))
    
def enhanceImage(img):
    """
    STw_ij = std deviation da janela centrada no pixel i,j
    Mw_ij  = mean da janela centrada no pixel i,j
    Iij    = intensidade do pixel (original) i,j
    W      = janela 48 x 36

    pixelMelhorado R = Funcao(STw_ij) * ( Iij - Mw_ij ) + Mw_ij
    """
    II = ImagemIntegral(img)
    height, width = img.shape[:2]
    enhanced = np.zeros( (height, width) )
    
    for i in xrange(height):
        for j in xrange(width):
            wW, wH = 48, 36
            x, y = i-(wW/2), j-(wH/2)
            
            if x < 0:
                wW += x   
                x = 0
            if y < 0:
                wH += y
                y = 0
            if x + wW >= height:
                wW -= x + wW - height
            if y + wH >= width:
                wH -= y + wH - width
            mean = II.media(x, y, wW, wH)
            stdDev = II.desvioPadrao(x, y, wW, wH)
            
            #print "i,j=(%s,%s) :: x,y=(%s,%s) :: mean=%s :: stdDev=%s :: enhanceCoef=%s" % (i,j,x,y,mean,stdDev,enhancementCoefficient(stdDev))
            #raw_input()
            enhanced[i,j] = enhancementCoefficient(stdDev)*(img[i,j] - mean) + mean
            #enhanced[i,j] = enhancementCoefficient(astd)*(img[i,j] - mean) + mean
    return enhanced
    
###################################################
def extractVerticalEdge(img):
    """
    2) Vertical Edge Extraction
                                                        | -1 0 1 
    convolui a imagem com o operador sobel vertical     | -2 0 2
    ||                                                  | -1 0 1
    para pegar a imagem gradiente vertical

    pega um threshold da imagem

    usa esse threshold e aplica nonmaximum supression na
    direção horizontal na imagem gradiente e temos 
    o resultado desse passo
    """
    pass
    
###################################################
def removeNoise(img):
    """
    3) Background Curve and Noise Removing 
    E = resultado do passo anterior (edge image)
    se pixel Pij for um ponto de aresta, então Eij = 1, caso contrário Eij = 0

    M e N matrizes de mesmo tamanho que E
    Tl e Ts (long/short): thresholds de tamanho de placa (padrao: Tl=28, Ts=5)

    A) inicializa M e N pra matrizes 0s
    B) 
    for each row i from top-to-bottom do
        for each column j from left-to-right do
            if (Ei,j == 1)
                if (Ei-1,j-1 + Ei-1,j + Ei-1,j+1 + Ei,j-1 > 0)
                    Mi,j = max{Mi-1,j-1, Mi-1,j, Mi-1,j+1, Mi,j-1} + 1;
                else
                    Mi,j = max{Mi-2,j-1, Mi-2,j, Mi-2,j+1, Mi-1,j-2,Mi-1,j+2,Mi,j-2} + 1;

    for each row i from bottom-to-top do
        for each column j from right-to-left do
            if (Ei,j == 1)
                if (Ei+1,j-1 + Ei+1,j + Ei+1,j+1+Ei,j+1 > 0)
                    Ni,j = max{Ni+1,j-1, Ni+1,j,Ni+1,j+1,Ni,j+1} + 1;
                else
                    Ni,j = max{Ni+2,j-1,Ni+2,j,Ni+2,j+1,Ni+1,j-2,Ni+1,j+2,Ni,j+2} + 1;

    for each row i from top-to-bottom do
        for each column j from left-to-right do
            if (Ei,j == 1)
                if (Mi,j + Ni,j > Tl || Mi,j + Ni,j < Ts)
                    Ei,j = 0;
    """
    pass

###################################################
def findPlate(img):
    """
    4) License Plate Search and Segmentation
    """
    pass
    
######################################################################################################
# Main
######################################################################################################

if __name__ == '__main__':
    import sys
    original = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    enhanced = enhanceImage(original)
    #cv2.imshow('Original', original)
    #cv2.imshow('Melhorada', enhanced)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    import pylab
    def PlotImage(image, title):
        pylab.imshow(image, cmap=pylab.cm.gray)
        pylab.title(title)
    pylab.subplot(121)
    PlotImage(original, "Original Image")
    pylab.subplot(122)
    PlotImage(enhanced, "Enhanced Image")
    pylab.show()