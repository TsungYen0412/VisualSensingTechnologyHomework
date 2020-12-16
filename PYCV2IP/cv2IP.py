#!/usr/bin/python3

import cv2
import numpy as np
import enum
from matplotlib import pyplot as plt
import math

class BaseIP(object):
    Obj_Num = 0

    def __init__(self):
        BaseIP.Obj_Num += 1
        print("Create 1 obj: Total number of BaseIP objects is "+ str(BaseIP.Obj_Num))

    def __del__(self):
        BaseIP.Obj_Num -= 1
        print("Delete 1 obj: Total number of BaseIP objects is "+ str(BaseIP.Obj_Num))

    @staticmethod
    def ImRead(filename):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def ImWrite(filename, img):
        cv2.imwrite(filename, img)

    @staticmethod
    def ImShow(winname, img):
        cv2.imshow(winname, img)

    @staticmethod
    def ImWindow(winname):
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)


class AlphaBlend(BaseIP):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def SplitAlpha(SrcImg):
        fore = cv2.merge([SrcImg[:,:,0], SrcImg[:,:,1], SrcImg[:,:,2]])
        alpha = cv2.merge([SrcImg[:,:,3], SrcImg[:,:,3], SrcImg[:,:,3]])
        return fore, alpha
    
    @staticmethod
    def DoBlending(Foreground, Background, Alpha):
        fore = Foreground * Alpha
        back = Background * (1.0 - Alpha)
        out = fore + back
        return out

    @staticmethod
    def MyDoBlending(Foreground, Background, Alpha, Beta):
        My_fore = AlphaBlend.DoBlending(Foreground * Alpha, Background * Alpha, Beta)
        My_back = Background * (1.0 - Alpha)
        My_out = My_fore + My_back
        return My_out


class HistIP(BaseIP):

    def __init__(self):
        super().__init__()

    class ColorType(enum.IntEnum):
        USE_RGB = 1
        USE_HSV = 2
        USE_YUV = 3

    @staticmethod
    def ImBGR2Gray(SrcImg):
        DstImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
        return DstImg

    @staticmethod
    def ImBGRA2BGR(SrcImg):
        DstImg = np.array(SrcImg[:,:,:3])
        return DstImg

    @staticmethod
    def CalcGrayHist(SrcGray):
        GrayHist = cv2.calcHist([SrcGray], [0], None, [256], [0, 256])
        return GrayHist

    @staticmethod
    def ShowGrayHist(winname, GrayHist):
        plt.plot(GrayHist, "gray")
        plt.legend(["Gray"], loc="upper left")
        plt.title(winname)
        plt.xlabel("Bins")
        plt.ylabel("Percentage of pixels")
        plt.show()

    @staticmethod
    def CalcColorHist(SrcColor):
        BlueHist = cv2.calcHist([SrcColor], [0], None, [256], [0, 256])
        GreenHist = cv2.calcHist([SrcColor], [1], None, [256], [0, 256])
        RedHist = cv2.calcHist([SrcColor], [2], None, [256], [0, 256])
        ColorHist = cv2.merge([BlueHist, GreenHist, RedHist])
        return ColorHist

    @staticmethod
    def ShowColorHist(winname, ColorHist):
        plt.plot(ColorHist[:,:,0], "b")
        plt.plot(ColorHist[:,:,1], "g")
        plt.plot(ColorHist[:,:,2], "r")
        plt.legend(["Blue", "Green", "Red"], loc="upper left")
        plt.title(winname)
        plt.xlabel("Bins")
        plt.ylabel("Percentage of pixels")
        plt.show()

    @staticmethod
    def MonoEqualize(SrcGray):
        EqualizedGray = cv2.equalizeHist(SrcGray)
        return EqualizedGray

    @staticmethod
    def ColorEqualize(SrcColor, CType = ColorType.USE_HSV):
        if CType == HistIP.ColorType.USE_RGB:
            EqualizedBlue = cv2.equalizeHist(SrcColor[:,:,0])
            EqualizedGreen = cv2.equalizeHist(SrcColor[:,:,1])
            EqualizedRed = cv2.equalizeHist(SrcColor[:,:,2])
            EqualizedColor = cv2.merge([EqualizedBlue, EqualizedGreen, EqualizedRed])
        elif CType == HistIP.ColorType.USE_HSV:
            SrcHSV = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2HSV)
            EqualizedHSV = np.array(SrcHSV)
            EqualizedHSV[:,:,2] = cv2.equalizeHist(SrcHSV[:,:,2])
            EqualizedColor = cv2.cvtColor(EqualizedHSV, cv2.COLOR_HSV2BGR)
        elif CType == HistIP.ColorType.USE_YUV:
            SrcYUV = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2YUV)
            EqualizedYUV = np.array(SrcYUV)
            EqualizedYUV[:,:,0] = cv2.equalizeHist(SrcYUV[:,:,0])
            EqualizedColor = cv2.cvtColor(EqualizedYUV, cv2.COLOR_YUV2BGR)
        return EqualizedColor

    @staticmethod
    def CalPDFGrayHist(SrcImg):
        GrayHist = HistIP.CalcGrayHist(SrcImg)
        PDFGrayHist = GrayHist / SrcImg.size
        return PDFGrayHist

    @staticmethod
    def CalPDFColorHist(SrcImg):
        ColorHist = HistIP.CalcColorHist(SrcImg)
        PDFColorHist = ColorHist / SrcImg[:,:,0].size
        return PDFColorHist

    @staticmethod
    def CalCDFGrayHist(SrcImg):
        PDFGrayHist = HistIP.CalPDFGrayHist(SrcImg)
        CDFGrayHist = np.zeros(PDFGrayHist.shape)
        CDFGrayHist[0,:] = PDFGrayHist[0,:]
        for i in range(1,256,1):
            CDFGrayHist[i,:] = CDFGrayHist[i-1,:] + PDFGrayHist[i,:]
        return CDFGrayHist

    @staticmethod
    def CalCDFColorHist(SrcImg):
        PDFColorHist = HistIP.CalPDFColorHist(SrcImg)
        CDFColorHist = np.zeros(PDFColorHist.shape)
        CDFColorHist[0,:,:] = PDFColorHist[0,:,:]
        for i in range(1,256,1):
            CDFColorHist[i,:,:] = CDFColorHist[i-1,:,:] + PDFColorHist[i,:,:]
        return CDFColorHist

    @staticmethod
    def MyCalculateLUT(SrcCDFHist, RefCDFHist):
        DiffCDFHist = np.zeros((256,256))
        for i in range(256):
            for j in range(256):
                DiffCDFHist[i][j] = math.fabs(SrcCDFHist[i] - RefCDFHist[j])
        LUT = np.zeros(256, dtype=np.int)
        for i in range(256):
            Min = DiffCDFHist[i][0]
            Index = 0
            for j in range(1,256,1):
                if DiffCDFHist[i][j] < Min:
                    Min = DiffCDFHist[i][j]
                    Index = j
            LUT[i] = Index
        return LUT

    @staticmethod
    def CalculateLUT(SrcCDFHist, RefCDFHist, Epsilon = 0.05):
        LUT = np.zeros(256, dtype=np.int)
        Last = 0
        for i in range(256):
            for j in range(Last,256,1):
                if abs(RefCDFHist[j,0] - SrcCDFHist[i,0]) < Epsilon or RefCDFHist[j,0] > SrcCDFHist[i,0]:
                    LUT[i] = j
                    Last = j
                    break
        return LUT

    @staticmethod
    def HistMatching(SrcImg, RefImg, Epsilon, CType = ColorType.USE_HSV):
        if CType == HistIP.ColorType.USE_RGB:
            #---------------------CDF---------------------#
            Src_CDFHist = HistIP.CalCDFColorHist(SrcImg)
            Ref_CDFHist = HistIP.CalCDFColorHist(RefImg)
            #---------------------LUT---------------------#
            BlueLUT = HistIP.CalculateLUT(Src_CDFHist[:,:,0], Ref_CDFHist[:,:,0], Epsilon)
            GreenLUT = HistIP.CalculateLUT(Src_CDFHist[:,:,1], Ref_CDFHist[:,:,1], Epsilon)
            RedLUT = HistIP.CalculateLUT(Src_CDFHist[:,:,2], Ref_CDFHist[:,:,2], Epsilon)
            LUT = cv2.merge([BlueLUT, GreenLUT, RedLUT])
            DstImg = np.array(SrcImg)
            for i in range(3):
                DstImg[:,:,i] = cv2.LUT(SrcImg[:,:,i], LUT[:,0,i])
        elif CType == HistIP.ColorType.USE_HSV:
            SrcHSV = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2HSV)
            RefHSV = cv2.cvtColor(RefImg, cv2.COLOR_BGR2HSV)
            #---------------------CDF---------------------#
            Src_CDFHist = HistIP.CalCDFGrayHist(SrcHSV[:,:,2])
            Ref_CDFHist = HistIP.CalCDFGrayHist(RefHSV[:,:,2])
            #---------------------LUT---------------------#
            LUT = HistIP.CalculateLUT(Src_CDFHist, Ref_CDFHist, Epsilon)
            DstHSV = np.array(SrcHSV)
            DstHSV[:,:,2] = cv2.LUT(SrcHSV[:,:,2], LUT)
            DstImg = cv2.cvtColor(DstHSV, cv2.COLOR_HSV2BGR)
        elif CType == HistIP.ColorType.USE_YUV:
            SrcYUV = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2YUV)
            RefYUV = cv2.cvtColor(RefImg, cv2.COLOR_BGR2YUV)
            #---------------------CDF---------------------#
            Src_CDFHist = HistIP.CalCDFGrayHist(SrcYUV[:,:,0])
            Ref_CDFHist = HistIP.CalCDFGrayHist(RefYUV[:,:,0])
            #---------------------LUT---------------------#
            LUT = HistIP.CalculateLUT(Src_CDFHist, Ref_CDFHist, Epsilon)
            DstYUV = np.array(SrcYUV)
            DstYUV[:,:,0] = cv2.LUT(SrcYUV[:,:,0], LUT, Epsilon)
            DstImg = cv2.cvtColor(DstYUV, cv2.COLOR_YUV2BGR)
        return DstImg


class ConvIP(BaseIP):

    def __init__(self):
        super().__init__()

    class SmoothType(enum.IntEnum):
        BLUR = 1
        BOX = 2
        GAUSSIAN = 3
        MEDIAN = 4
        BILATERAL = 5

    class EdgeType(enum.IntEnum):
        SOBEL = 1
        CANNY = 2
        SCHARR = 3
        LAPLACE = 4
        COLOR_SOBEL = 5

    @staticmethod
    def Smooth2D(SrcImg, ksize = 15, SmType = SmoothType.BLUR):
        if SmType == ConvIP.SmoothType.BLUR:
            OutImg = cv2.blur(SrcImg, (ksize, ksize))
        elif SmType == ConvIP.SmoothType.BOX:
            OutImg = cv2.boxFilter(SrcImg, -1, (ksize, ksize))
        elif SmType == ConvIP.SmoothType.GAUSSIAN:
            OutImg = cv2.GaussianBlur(SrcImg, (ksize, ksize), 0)
        elif SmType == ConvIP.SmoothType.MEDIAN:
            OutImg = cv2.medianBlur(SrcImg, ksize)
        elif SmType == ConvIP.SmoothType.BILATERAL:
            OutImg = cv2.bilateralFilter(SrcImg, 9, 75, 75)
        return OutImg

    @staticmethod
    def EdgeDetect(SrcImg, EdType = EdgeType.SOBEL):
        if EdType == ConvIP.EdgeType.SOBEL:
            GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            Gradient_X_64F = cv2.Sobel(GrayImg, cv2.CV_64F, 1, 0)
            Gradient_Y_64F = cv2.Sobel(GrayImg, cv2.CV_64F, 0, 1)
            Gradient_X_8U = cv2.convertScaleAbs(Gradient_X_64F)
            Gradient_Y_8U = cv2.convertScaleAbs(Gradient_Y_64F)
            OutImg = cv2.convertScaleAbs(Gradient_X_8U * 0.5 + Gradient_Y_8U * 0.5)
        elif EdType == ConvIP.EdgeType.CANNY:
            GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            BlurredImg = cv2.GaussianBlur(GrayImg, (3, 3), 0)
            OutImg = cv2.Canny(BlurredImg, 30, 70)
        elif EdType == ConvIP.EdgeType.SCHARR:
            GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            Gradient_X_64F = cv2.Scharr(GrayImg, cv2.CV_64F, 1, 0)
            Gradient_Y_64F = cv2.Scharr(GrayImg, cv2.CV_64F, 0, 1)
            Gradient_X_8U = cv2.convertScaleAbs(Gradient_X_64F)
            Gradient_Y_8U = cv2.convertScaleAbs(Gradient_Y_64F)
            OutImg = cv2.convertScaleAbs(Gradient_X_8U * 0.5 + Gradient_Y_8U * 0.5)
        elif EdType == ConvIP.EdgeType.LAPLACE:
            GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            OutImg_64F = cv2.Laplacian(GrayImg, cv2.CV_64F, ksize = 3)
            OutImg = cv2.convertScaleAbs(OutImg_64F)
        elif EdType == ConvIP.EdgeType.COLOR_SOBEL:
            Gradient_X_64F = cv2.Sobel(SrcImg, cv2.CV_64F, 1, 0)
            Gradient_Y_64F = cv2.Sobel(SrcImg, cv2.CV_64F, 0, 1)
            Gradient_X_8U = cv2.convertScaleAbs(Gradient_X_64F)
            Gradient_Y_8U = cv2.convertScaleAbs(Gradient_Y_64F)
            OutImg = cv2.convertScaleAbs(Gradient_X_8U * 0.5 + Gradient_Y_8U * 0.5)
        return OutImg
