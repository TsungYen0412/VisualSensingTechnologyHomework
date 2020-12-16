#!/usr/bin/python3

import cv2
import numpy as np
import cv2IP
import tkinter as tk

def MyShowGrayImage():
    IP = cv2IP.HistIP()
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    if SrcImg.shape[2] == 4:
        F_BGR = IP.ImBGRA2BGR(SrcImg)
    else:
        F_BGR = np.array(SrcImg)
    F_Gray = IP.ImBGR2Gray(F_BGR)
    IP.ImWindow("ForeGround Gray Image")
    IP.ImShow("ForeGround Gray Image", F_Gray)
    cv2.waitKey(0)
    del IP


def Example_ImSmooth(SmType):
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/InputIm_1_FixdPoint.bmp")
    IP.ImShow("Original Image", SrcImg)

    OutImg = IP.Smooth2D(SrcImg, 5, SmType)
    IP.ImShow("Smoothed Color Image - 5", OutImg)

    OutImg = IP.Smooth2D(SrcImg, 15, SmType)
    IP.ImShow("Smoothed Color Image - 15", OutImg)
    cv2.waitKey(0)
    del IP


def Example_ImEdge(EdType):
    IP = cv2IP.ConvIP()
    # source image
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    IP.ImShow("Original Image", SrcImg)
    if SrcImg.shape[2] == 4:
        Src_BGR = np.array(SrcImg[:,:,:3])
    else:
        Src_BGR = np.array(SrcImg)

    OutImg = IP.EdgeDetect(Src_BGR, EdType)

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


if __name__ == '__main__':
    # Example_ImSmooth(cv2IP.ConvIP.SmoothType.BILATERAL)
    Example_ImEdge(cv2IP.ConvIP.EdgeType.CANNY)
