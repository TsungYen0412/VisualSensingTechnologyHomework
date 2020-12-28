#!/usr/bin/python3

import cv2
import numpy as np
import cv2IP
import tkinter as tk

def Example_ImSmooth(SmType):
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/InputIm_1_FixdPoint.bmp")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        SrcBGR = np.array(SrcImg[:,:,:3])
    else:
        SrcBGR = np.array(SrcImg)

    OutImg = IP.Smooth2D(SrcBGR, 5, SmType)
    IP.ImShow("Smoothed Color Image - 5", OutImg)

    OutImg = IP.Smooth2D(SrcBGR, 15, SmType)
    IP.ImShow("Smoothed Color Image - 15", OutImg)
    cv2.waitKey(0)
    del IP


def Example_ImEdge(EdType):
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        SrcBGR = np.array(SrcImg[:,:,:3])
    else:
        SrcBGR = np.array(SrcImg)

    OutImg = IP.EdgeDetect(SrcBGR, EdType)

    if EdType == IP.EdgeType.SOBEL:
        IP.ImShow("Sobel Edge Image", OutImg)
    elif EdType == IP.EdgeType.CANNY:
        IP.ImShow("Canny Edge Image", OutImg)
    elif EdType == IP.EdgeType.SCHARR:
        IP.ImShow("SCHARR Edge Image", OutImg)
    elif EdType == IP.EdgeType.LAPLACE:
        IP.ImShow("LAPLACE Edge Image", OutImg)
    elif EdType == IP.EdgeType.COLOR_SOBEL:
        IP.ImShow("COLOR SOBEL Edge Image", OutImg)
    cv2.waitKey(0)
    del IP

def Example_ImConv2D_Roberts():
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        SrcBGR = np.array(SrcImg[:,:,:3])
    else:
        SrcBGR = np.array(SrcImg)

    SrcGray = IP.ImBGR2Gray(SrcBGR)
    Kernels = IP.GetRobertsKernel()
    Grad_Planes = []
    for i in range(0, len(Kernels)):
        Grad_Planes.append(IP.Conv2D(SrcGray, Kernels[i]))
        Grad_Planes[i] = cv2.convertScaleAbs(Grad_Planes[i])

    GradImg = Grad_Planes[0] * 0.5
    for i in range(1, len(Kernels)):
        GradImg += Grad_Planes[i] * 0.5
    GradImg = cv2.convertScaleAbs(GradImg)
    IP.ImShow("Roberts Image", GradImg)
    cv2.waitKey(0)
    del IP

def Example_ImConv2D_Prewitt():
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        SrcBGR = np.array(SrcImg[:,:,:3])
    else:
        SrcBGR = np.array(SrcImg)

    SrcGray = IP.ImBGR2Gray(SrcBGR)
    Kernels = IP.GetPrewittKernel()
    Grad_Planes = []
    for i in range(0, len(Kernels)):
        Grad_Planes.append(IP.Conv2D(SrcGray, Kernels[i]))
        Grad_Planes[i] = cv2.convertScaleAbs(Grad_Planes[i])

    GradImg = Grad_Planes[0] * 0.5
    for i in range(1, len(Kernels)):
        GradImg += Grad_Planes[i] * 0.5
    GradImg = cv2.convertScaleAbs(GradImg)
    IP.ImShow("Prewitt Image", GradImg)
    cv2.waitKey(0)
    del IP

def Example_ImConv2D_Kirsch():
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        SrcBGR = np.array(SrcImg[:,:,:3])
    else:
        SrcBGR = np.array(SrcImg)

    SrcGray = IP.ImBGR2Gray(SrcBGR)
    Kernels = IP.GetKirschKernel()
    Grad_Planes = []
    for i in range(0, len(Kernels)):
        Grad_Planes.append(IP.Conv2D(SrcGray, Kernels[i]))
        Grad_Planes[i] = cv2.convertScaleAbs(Grad_Planes[i])

    GradImg = Grad_Planes[0] * 0.5
    for i in range(1, len(Kernels)):
        GradImg += Grad_Planes[i] * 0.5
    GradImg = cv2.convertScaleAbs(GradImg)
    IP.ImShow("Kirsch Image", GradImg)
    cv2.waitKey(0)
    del IP

def Example_ImSharpening(SpType, SmType):
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        SrcBGR = np.array(SrcImg[:,:,:3])
    else:
        SrcBGR = np.array(SrcImg)

    DstImg = IP.ImSharpening(SrcImg, SpType, SmType)
    IP.ImShow("Sharpening Image", DstImg)
    cv2.waitKey(0)
    del IP

if __name__ == '__main__':
    # Example_ImSmooth(cv2IP.ConvIP.SmoothType.BILATERAL)
    # Example_ImEdge(cv2IP.ConvIP.EdgeType.COLOR_SOBEL)
    # Example_ImConv2D_Kirsch()
    Example_ImSharpening(cv2IP.ConvIP.SharpType.UNSHARP_MASK, cv2IP.ConvIP.SmoothType.GAUSSIAN)
