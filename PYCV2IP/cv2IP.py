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

    def ImRead(self, filename):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    def ImWrite(self, filename, img):
        cv2.imwrite(filename, img)

    def ImShow(self, winname, img):
        cv2.imshow(winname, img)

    def ImWindow(self, winname):
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    def ImBGR2Gray(self, SrcImg):
        DstImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
        return DstImg

    def ImBGRA2BGR(self, SrcImg):
        DstImg = np.array(SrcImg[:,:,:3])
        return DstImg


class AlphaBlend(BaseIP):

    def __init__(self):
        super().__init__()
    
    def SplitAlpha(self, SrcImg):
        fore = cv2.merge([SrcImg[:,:,0], SrcImg[:,:,1], SrcImg[:,:,2]])
        alpha = cv2.merge([SrcImg[:,:,3], SrcImg[:,:,3], SrcImg[:,:,3]])
        return fore, alpha

    def DoBlending(self, Foreground, Background, Alpha):
        fore = Foreground * Alpha
        back = Background * (1.0 - Alpha)
        out = fore + back
        return out

    def MyDoBlending(self, Foreground, Background, Alpha, Beta):
        My_fore = self.DoBlending(Foreground * Alpha, Background * Alpha, Beta)
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

    def CalcGrayHist(self, SrcGray):
        GrayHist = cv2.calcHist([SrcGray], [0], None, [256], [0, 256])
        return GrayHist

    def ShowGrayHist(self, winname, GrayHist):
        plt.plot(GrayHist, "gray")
        plt.legend(["Gray"], loc="upper left")
        plt.title(winname)
        plt.xlabel("Bins")
        plt.ylabel("Percentage of pixels")
        plt.show()

    def CalcColorHist(self, SrcColor):
        BlueHist = cv2.calcHist([SrcColor], [0], None, [256], [0, 256])
        GreenHist = cv2.calcHist([SrcColor], [1], None, [256], [0, 256])
        RedHist = cv2.calcHist([SrcColor], [2], None, [256], [0, 256])
        ColorHist = cv2.merge([BlueHist, GreenHist, RedHist])
        return ColorHist

    def ShowColorHist(self, winname, ColorHist):
        plt.plot(ColorHist[:,:,0], "b")
        plt.plot(ColorHist[:,:,1], "g")
        plt.plot(ColorHist[:,:,2], "r")
        plt.legend(["Blue", "Green", "Red"], loc="upper left")
        plt.title(winname)
        plt.xlabel("Bins")
        plt.ylabel("Percentage of pixels")
        plt.show()

    def MonoEqualize(self, SrcGray):
        EqualizedGray = cv2.equalizeHist(SrcGray)
        return EqualizedGray

    def ColorEqualize(self, SrcColor, CType = ColorType.USE_HSV):
        if CType == self.ColorType.USE_RGB:
            EqualizedBlue = cv2.equalizeHist(SrcColor[:,:,0])
            EqualizedGreen = cv2.equalizeHist(SrcColor[:,:,1])
            EqualizedRed = cv2.equalizeHist(SrcColor[:,:,2])
            EqualizedColor = cv2.merge([EqualizedBlue, EqualizedGreen, EqualizedRed])
        elif CType == self.ColorType.USE_HSV:
            SrcHSV = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2HSV)
            EqualizedHSV = np.array(SrcHSV)
            EqualizedHSV[:,:,2] = cv2.equalizeHist(SrcHSV[:,:,2])
            EqualizedColor = cv2.cvtColor(EqualizedHSV, cv2.COLOR_HSV2BGR)
        elif CType == self.ColorType.USE_YUV:
            SrcYUV = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2YUV)
            EqualizedYUV = np.array(SrcYUV)
            EqualizedYUV[:,:,0] = cv2.equalizeHist(SrcYUV[:,:,0])
            EqualizedColor = cv2.cvtColor(EqualizedYUV, cv2.COLOR_YUV2BGR)
        return EqualizedColor

    def CalPDFGrayHist(self, SrcImg):
        GrayHist = self.CalcGrayHist(SrcImg)
        PDFGrayHist = GrayHist / SrcImg.size
        return PDFGrayHist

    def CalPDFColorHist(self, SrcImg):
        ColorHist = self.CalcColorHist(SrcImg)
        PDFColorHist = ColorHist / SrcImg[:,:,0].size
        return PDFColorHist

    def CalCDFGrayHist(self, SrcImg):
        PDFGrayHist = self.CalPDFGrayHist(SrcImg)
        CDFGrayHist = np.zeros(PDFGrayHist.shape)
        CDFGrayHist[0,:] = PDFGrayHist[0,:]
        for i in range(1,256,1):
            CDFGrayHist[i,:] = CDFGrayHist[i-1,:] + PDFGrayHist[i,:]
        return CDFGrayHist

    def CalCDFColorHist(self, SrcImg):
        PDFColorHist = self.CalPDFColorHist(SrcImg)
        CDFColorHist = np.zeros(PDFColorHist.shape)
        CDFColorHist[0,:,:] = PDFColorHist[0,:,:]
        for i in range(1,256,1):
            CDFColorHist[i,:,:] = CDFColorHist[i-1,:,:] + PDFColorHist[i,:,:]
        return CDFColorHist

    def MyCalculateLUT(self, SrcCDFHist, RefCDFHist):
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

    def CalculateLUT(self, SrcCDFHist, RefCDFHist, Epsilon = 0.05):
        LUT = np.zeros(256, dtype=np.int)
        Last = 0
        for i in range(256):
            for j in range(Last,256,1):
                if abs(RefCDFHist[j,0] - SrcCDFHist[i,0]) < Epsilon or RefCDFHist[j,0] > SrcCDFHist[i,0]:
                    LUT[i] = j
                    Last = j
                    break
        return LUT

    def HistMatching(self, SrcImg, RefImg, Epsilon, CType = ColorType.USE_HSV):
        if CType == self.ColorType.USE_RGB:
            #---------------------CDF---------------------#
            Src_CDFHist = self.CalCDFColorHist(SrcImg)
            Ref_CDFHist = self.CalCDFColorHist(RefImg)
            #---------------------LUT---------------------#
            BlueLUT = self.CalculateLUT(Src_CDFHist[:,:,0], Ref_CDFHist[:,:,0], Epsilon)
            GreenLUT = self.CalculateLUT(Src_CDFHist[:,:,1], Ref_CDFHist[:,:,1], Epsilon)
            RedLUT = self.CalculateLUT(Src_CDFHist[:,:,2], Ref_CDFHist[:,:,2], Epsilon)
            LUT = cv2.merge([BlueLUT, GreenLUT, RedLUT])
            DstImg = np.array(SrcImg)
            for i in range(3):
                DstImg[:,:,i] = cv2.LUT(SrcImg[:,:,i], LUT[:,0,i])
        elif CType == self.ColorType.USE_HSV:
            SrcHSV = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2HSV)
            RefHSV = cv2.cvtColor(RefImg, cv2.COLOR_BGR2HSV)
            #---------------------CDF---------------------#
            Src_CDFHist = self.CalCDFGrayHist(SrcHSV[:,:,2])
            Ref_CDFHist = self.CalCDFGrayHist(RefHSV[:,:,2])
            #---------------------LUT---------------------#
            LUT = self.CalculateLUT(Src_CDFHist, Ref_CDFHist, Epsilon)
            DstHSV = np.array(SrcHSV)
            DstHSV[:,:,2] = cv2.LUT(SrcHSV[:,:,2], LUT)
            DstImg = cv2.cvtColor(DstHSV, cv2.COLOR_HSV2BGR)
        elif CType == self.ColorType.USE_YUV:
            SrcYUV = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2YUV)
            RefYUV = cv2.cvtColor(RefImg, cv2.COLOR_BGR2YUV)
            #---------------------CDF---------------------#
            Src_CDFHist = self.CalCDFGrayHist(SrcYUV[:,:,0])
            Ref_CDFHist = self.CalCDFGrayHist(RefYUV[:,:,0])
            #---------------------LUT---------------------#
            LUT = self.CalculateLUT(Src_CDFHist, Ref_CDFHist, Epsilon)
            DstYUV = np.array(SrcYUV)
            DstYUV[:,:,0] = cv2.LUT(SrcYUV[:,:,0], LUT, Epsilon)
            DstImg = cv2.cvtColor(DstYUV, cv2.COLOR_YUV2BGR)
        return DstImg


class ConvIP(BaseIP):

    def __init__(self):
        super().__init__()
        self.__RobertsKernel = np.zeros((2, 2, 2), dtype=np.int) # Gx, Gy
        self.__PrewittKernel = np.zeros((3, 3, 2), dtype=np.int) # x, y
        self.__KirschKernel = np.full((3, 3, 8), -3, dtype=np.int) # E, NE, N, NW, W, SW, S, SE

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

    class SharpType(enum.IntEnum):
        LAPLACE_TYPE1 = 1
        LAPLACE_TYPE2 = 2
        SECOND_ORDER_LOG = 3
        UNSHARP_MASK = 4

    def Smooth2D(self, SrcImg, ksize = 15, SmType = SmoothType.BLUR,
                  d = 9, sigma = 75):
        if SmType == self.SmoothType.BLUR:
            OutImg = cv2.blur(SrcImg, (ksize, ksize))
        elif SmType == self.SmoothType.BOX:
            OutImg = cv2.boxFilter(SrcImg, -1, (ksize, ksize))
        elif SmType == self.SmoothType.GAUSSIAN:
            OutImg = cv2.GaussianBlur(SrcImg, (ksize, ksize), 0)
        elif SmType == self.SmoothType.MEDIAN:
            OutImg = cv2.medianBlur(SrcImg, ksize)
        elif SmType == self.SmoothType.BILATERAL:
            OutImg = cv2.bilateralFilter(SrcImg, d, sigma, sigma)
        return OutImg

    def EdgeDetect(self, SrcImg, EdType = EdgeType.SOBEL, ksize = 3,
                    lowThreshold = 80, highThreshold = 150):
        if EdType == self.EdgeType.SOBEL:
            GrayImg = self.ImBGR2Gray(SrcImg)
            Gradient_X_64F = cv2.Sobel(GrayImg, cv2.CV_64F, 1, 0, ksize = ksize)
            Gradient_Y_64F = cv2.Sobel(GrayImg, cv2.CV_64F, 0, 1, ksize = ksize)
            Gradient_X_8U = cv2.convertScaleAbs(Gradient_X_64F)
            Gradient_Y_8U = cv2.convertScaleAbs(Gradient_Y_64F)
            OutImg = cv2.convertScaleAbs(Gradient_X_8U * 0.5 + Gradient_Y_8U * 0.5)
        elif EdType == self.EdgeType.CANNY:
            GrayImg = self.ImBGR2Gray(SrcImg)
            BlurredImg = cv2.GaussianBlur(GrayImg, (ksize, ksize), 0)
            OutImg = cv2.Canny(BlurredImg, lowThreshold, highThreshold)
        elif EdType == self.EdgeType.SCHARR:
            GrayImg = self.ImBGR2Gray(SrcImg)
            Gradient_X_64F = cv2.Scharr(GrayImg, cv2.CV_64F, 1, 0)
            Gradient_Y_64F = cv2.Scharr(GrayImg, cv2.CV_64F, 0, 1)
            Gradient_X_8U = cv2.convertScaleAbs(Gradient_X_64F)
            Gradient_Y_8U = cv2.convertScaleAbs(Gradient_Y_64F)
            OutImg = cv2.convertScaleAbs(Gradient_X_8U * 0.5 + Gradient_Y_8U * 0.5)
        elif EdType == self.EdgeType.LAPLACE:
            GrayImg = self.ImBGR2Gray(SrcImg)
            OutImg_64F = cv2.Laplacian(GrayImg, cv2.CV_64F, ksize = ksize)
            OutImg = cv2.convertScaleAbs(OutImg_64F)
        elif EdType == self.EdgeType.COLOR_SOBEL:
            Gradient_X_64F = cv2.Sobel(SrcImg, cv2.CV_64F, 1, 0, ksize = ksize)
            Gradient_Y_64F = cv2.Sobel(SrcImg, cv2.CV_64F, 0, 1, ksize = ksize)
            Gradient_X_8U = cv2.convertScaleAbs(Gradient_X_64F)
            Gradient_Y_8U = cv2.convertScaleAbs(Gradient_Y_64F)
            OutImg = cv2.convertScaleAbs(Gradient_X_8U * 0.5 + Gradient_Y_8U * 0.5)
        return OutImg

    def __InitRobertsKernel(self):
        #----------------Gx---------------#
        self.__RobertsKernel[0,0,0] =  1
        self.__RobertsKernel[1,1,0] = -1
        #----------------Gy---------------#
        self.__RobertsKernel[1,0,1] =  1
        self.__RobertsKernel[0,1,1] = -1

    def __InitPrewittKernel(self):
        #----------------x----------------#
        self.__PrewittKernel[0,0,0] = -1
        self.__PrewittKernel[2,0,0] =  1
        self.__PrewittKernel[0,1,0] = -1
        self.__PrewittKernel[2,1,0] =  1
        self.__PrewittKernel[0,2,0] = -1
        self.__PrewittKernel[2,2,0] =  1
        #----------------y----------------#
        self.__PrewittKernel[0,0,1] = -1
        self.__PrewittKernel[1,0,1] = -1
        self.__PrewittKernel[2,0,1] = -1
        self.__PrewittKernel[0,2,1] =  1
        self.__PrewittKernel[1,2,1] =  1
        self.__PrewittKernel[2,2,1] =  1
    
    def __InitKirschKernel(self):
        #----------------E----------------#
        self.__KirschKernel[2,0,0] = 5
        self.__KirschKernel[1,1,0] = 0
        self.__KirschKernel[2,1,0] = 5
        self.__KirschKernel[2,2,0] = 5
        #----------------NE---------------#
        self.__KirschKernel[1,0,1] = 5
        self.__KirschKernel[2,0,1] = 5
        self.__KirschKernel[1,1,1] = 0
        self.__KirschKernel[2,1,1] = 5
        #----------------N----------------#
        self.__KirschKernel[0,0,2] = 5
        self.__KirschKernel[1,0,2] = 5
        self.__KirschKernel[2,0,2] = 5
        self.__KirschKernel[1,1,2] = 0
        #----------------NW---------------#
        self.__KirschKernel[0,0,3] = 5
        self.__KirschKernel[1,0,3] = 5
        self.__KirschKernel[0,1,3] = 5
        self.__KirschKernel[1,1,3] = 0
        #----------------W----------------#
        self.__KirschKernel[0,0,4] = 5
        self.__KirschKernel[0,1,4] = 5
        self.__KirschKernel[1,1,4] = 0
        self.__KirschKernel[0,2,4] = 5
        #----------------SW---------------#
        self.__KirschKernel[0,1,5] = 5
        self.__KirschKernel[1,1,5] = 0
        self.__KirschKernel[0,2,5] = 5
        self.__KirschKernel[1,2,5] = 5
        #----------------S----------------#
        self.__KirschKernel[1,1,6] = 0
        self.__KirschKernel[0,2,6] = 5
        self.__KirschKernel[1,2,6] = 5
        self.__KirschKernel[2,2,6] = 5
        #----------------SE---------------#
        self.__KirschKernel[1,1,7] = 0
        self.__KirschKernel[2,1,7] = 5
        self.__KirschKernel[1,2,7] = 5
        self.__KirschKernel[2,2,7] = 5

    def GetRobertsKernel(self):
        self.__InitRobertsKernel()
        return (self.__RobertsKernel[:,:,0], self.__RobertsKernel[:,:,1])

    def GetPrewittKernel(self):
        self.__InitPrewittKernel()
        return (self.__PrewittKernel[:,:,0], self.__PrewittKernel[:,:,1])
    
    def GetKirschKernel(self):
        self.__InitKirschKernel()
        return (self.__KirschKernel[:,:,0], self.__KirschKernel[:,:,1], \
                self.__KirschKernel[:,:,2], self.__KirschKernel[:,:,3], \
                self.__KirschKernel[:,:,4], self.__KirschKernel[:,:,5], \
                self.__KirschKernel[:,:,6], self.__KirschKernel[:,:,7])

    def Conv2D(self, SrcImg, Kernel):
        DstImg = cv2.filter2D(SrcImg, -1, Kernel)
        return DstImg

    def ImSharpening(self, SrcImg, SpType = SharpType.UNSHARP_MASK, SmType = SmoothType.BILATERAL,
                      ksize = 3, d = 9, sigma = 75, Landa = 0.5):
        if SpType == self.SharpType.LAPLACE_TYPE1:
            Original = np.zeros((3, 3), dtype=np.int)
            Original[1,1] =  1
            Filtered = np.zeros((3, 3), dtype=np.int)
            Filtered[1,0] = -1
            Filtered[0,1] = -1
            Filtered[1,1] =  4
            Filtered[2,1] = -1
            Filtered[1,2] = -1
            Resulting = Original + Landa * Filtered
            DstImg = self.Conv2D(SrcImg, Resulting)
        elif SpType == self.SharpType.LAPLACE_TYPE2:
            Original = np.zeros((3, 3), dtype=np.int)
            Original[1,1] = 1
            Filtered = np.full((3, 3), -1, dtype=np.int)
            Filtered[1,1] = 8
            Resulting = Original + Landa * Filtered
            DstImg = self.Conv2D(SrcImg, Resulting)
        elif SpType == self.SharpType.SECOND_ORDER_LOG:
            Original = np.zeros((5, 5), dtype=np.int)
            Original[2,2] = 1
            Filtered = np.zeros((5, 5), dtype=np.int)
            Filtered[2,0] = -1
            Filtered[1,1] = -1
            Filtered[2,1] = -2
            Filtered[3,1] = -1
            Filtered[0,2] = -1
            Filtered[1,2] = -2
            Filtered[2,2] = 16
            Filtered[3,2] = -2
            Filtered[4,2] = -1
            Filtered[1,3] = -1
            Filtered[2,3] = -2
            Filtered[3,3] = -1
            Filtered[2,4] = -1
            Resulting = Original + Landa * Filtered
            DstImg = self.Conv2D(SrcImg, Resulting)
        elif SpType == self.SharpType.UNSHARP_MASK:
            Coarse = self.Smooth2D(SrcImg, ksize, SmType, d, sigma)
            Fine = SrcImg * 1.0 - Coarse * 1.0
            DstImg = cv2.convertScaleAbs(SrcImg + Landa * Fine)
        return DstImg
