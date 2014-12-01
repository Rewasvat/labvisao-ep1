#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from time import time

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
    enhanced = np.zeros( (height, width), dtype=img.dtype )
    
    for i in xrange(height):
        for j in xrange(width):
            wW, wH = 36, 48
            x, y = i-(wW/2), j-(wH/2)
            # corrigindo valores da janela pra não sair da imagem
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
            enhanced[i,j] = np.around( enhancementCoefficient(stdDev)*(img[i,j] - mean) + mean )
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
    #sobel = np.asarray( [ [-1,0,1], [-2,0,2], [-1,0,1] ] )
    #gradiente = cv2.filter2D(img,-1,sobel)
    gradiente = np.abs( cv2.Sobel(img,-1,1,0,ksize=3) )  #sobel pode retornar pixels negativos, só magnitude interessa aqui
    
    thresh, gradiente = cv2.threshold(np.abs(gradiente), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #print "thresh =", thresh
    
    # non-maximum suppression (horizontal direction)
    height, width = img.shape[:2]
    for i in xrange(height):
        seqStart, seqEnd = -1, -1
        for j in xrange(width):
            if gradiente[i,j] == 0 and seqStart == -1:
                seqStart = j
            elif gradiente[i,j] != 0 and seqEnd == -1 and seqStart != -1:
                seqEnd = j
                
                seq = img[i,seqStart:seqEnd]
                maxindex = seqStart + np.argmax(seq)
                gradiente[i, seqStart:seqEnd] = 255
                gradiente[i, maxindex] = 0
                
                seqStart, seqEnd = -1, -1
       
            
    return gradiente
    
###################################################
def removeNoise(imgo, longT, shortT):
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
    img = imgo.copy()
    height, width = img.shape[:2]
    def edge(i, j):
        if i < 0 or j < 0 or i >= height or j >= width:
            return 0
        return int( img[i,j] == 0 )

    M = np.zeros( (height+2, width+2) )
    N = np.zeros( (height+2, width+2) )
    Tl, Ts = longT, shortT  ##>> limites de tamanho pra limpar arestas

    for i in xrange(height):
        for j in xrange(width):
            if edge(i,j):
                if edge(i-1,j-1) + edge(i-1,j) + edge(i-1,j+1) + edge(i,j-1) > 0:
                    M[i,j] = max(M[i-1,j-1], M[i-1,j], M[i-1,j+1], M[i,j-1]) + 1
                else:
                    M[i,j] = max(M[i-2,j-1], M[i-2,j], M[i-2,j+1], M[i-1,j-2], M[i-1,j+2], M[i,j-2]) + 1

    for i in xrange(height-1,-1,-1):
        for j in xrange(width-1,-1,-1):
            if edge(i,j):
                if edge(i+1,j-1) + edge(i+1,j) + edge(i+1,j+1) + edge(i,j+1) > 0:
                    N[i,j] = max(N[i+1,j-1], N[i+1,j], N[i+1,j+1], N[i,j+1]) + 1;
                else:
                    N[i,j] = max(N[i+2,j-1], N[i+2,j], N[i+2,j+1], N[i+1,j-2], N[i+1,j+2], N[i,j+2]) + 1;

    for i in xrange(height):
        for j in xrange(width):
            if edge(i,j):
                if M[i,j] + N[i,j] > Tl or M[i,j] + N[i,j] < Ts:
                    #Ei,j = 0;
                    img[i,j] = 255

    return img
                    
###################################################
def findPlate(img, width, height):
    """
    4) License Plate Search and Segmentation
    """
    II = ImagemIntegral(img)
    
    wW, wH = height, width            ##>> tamanho da area pra procurar placa
    height, width = img.shape[:2]
    
    edgeTotal = -1
    wX, wY = -1,-1
    
    for i in xrange( (wW/2), height-(wW/2)):
        for j in xrange( (wH/2), width-(wH/2)):
            x, y = i-(wW/2), j-(wH/2)
            etot = II.area(x, y, wW, wH)
            if edgeTotal == -1 or etot < edgeTotal:
                edgeTotal, wX, wY = etot, x, y
          
    #print "total da placa=", II.area(96, 372, wW, wH)
    #print "total da area retornada=", edgeTotal
    return (wX, wY, wW, wH)
    
######################################################################################################
# Main
######################################################################################################
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze given image to find license plate location. The located sub-image of the plate will be saved as the same extension" +
        " of the original file, and the same name with a '_plate' suffix.\nExample:\n\tinput filename: someImage.png\n\toutput: someImage_plate.png")
    parser.add_argument("filename", help="Filename of the image to find a license plate in.")
    parser.add_argument("--enhance", "-e", action="store_true", default=False, help="Enhance the image before analizing it (default is False)")
    parser.add_argument("--display", "-d", action="store_true", default=False, help="Opens a pylab window to show the resulting image of each step. Pressing 'q' closes it" +
        " (default is False)")
    parser.add_argument("--longEdge", "-le", type=int, default=28, help="Threshold for long edges to be filtered from image. (default is 28)")
    parser.add_argument("--shortEdge", "-se", type=int, default=5, help="Threshold for short edges to be filtered from image. (default is 5)")
    parser.add_argument("--plateWidth", "-pw", type=int, default=150, help="Width of the window used to search for a plate. Should be a little bigger than the actual plate width" +
        " (default is 150)")
    parser.add_argument("--plateHeight", "-ph", type=int, default=60, help="Height of the window used to search for a plate. Should be a little bigger than the actual plate " + 
        "height (default is 60)")

    args = parser.parse_args()
    
    mark = time()
    total = 0.0
    original = cv2.imread(args.filename, 0)
    stepTime = time()-mark
    total += stepTime
    print "Imagem %s %s lida em %.2f secs." % (args.filename, original.shape, stepTime)
    
    if args.enhance:
        mark = time()
        enhanced = enhanceImage(original)
        stepTime = time()-mark
        total += stepTime
        print "Imagem melhorada em %.2f secs." % (stepTime)
    else:
        enhanced = original
    
    mark = time()
    edges = extractVerticalEdge(enhanced)
    stepTime = time()-mark
    total += stepTime
    print "Linhas verticais extraidas em %.2f secs." % (stepTime)
    
    mark = time()
    elimpa = removeNoise(edges, args.longEdge, args.shortEdge)
    stepTime = time()-mark
    total += stepTime
    print "Imagem limpa e ruidos removidos em %.2f secs." % (stepTime)
    
    mark = time()
    pX, pY, pW, pH = findPlate(elimpa, args.plateWidth, args.plateHeight)
    stepTime = time()-mark
    total += stepTime
    print "Subimagem da placa (%s, %s, %s, %s) localizada em %.2f secs." % (pX, pY, pW, pH, stepTime)
    placa = original[pX:pX+pW, pY:pY+pH]
    
    print "Full execution took %.2f secs" % (total) 
    
    outName = args.filename.split(".")
    outName = ".".join(outName[:-1]) + "_plate." + outName[-1]
    cv2.imwrite(outName, placa)
    print "Saved plate image to:", outName
    
    if args.display:
        import pylab
        def PlotImage(image, title, npos):
            pylab.subplot(npos)
            pylab.imshow(image, cmap=pylab.cm.gray)
            pylab.title(title)
        PlotImage(original, "Original Image",               231)
        PlotImage(enhanced, "Enhanced Image",               232)
        PlotImage(edges,    "Edges (Gradient Image)",       233)
        PlotImage(elimpa,   "Edges (Limpa, sem ruidos)",    234)
        PlotImage(placa,    "Subimagem da Placa",           235)
        
        def onPress(event):
            if event.key == 'q':
                pylab.close()
        pylab.connect('key_press_event', onPress)
        pylab.show()
        
    
