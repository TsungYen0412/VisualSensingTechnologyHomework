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

def MyShowGrayHistogram():
    IP = cv2IP.HistIP()
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    if SrcImg.shape[2] == 4:
        F_BGR = IP.ImBGRA2BGR(SrcImg)
    else:
        F_BGR = np.array(SrcImg)
    F_Gray = IP.ImBGR2Gray(F_BGR)
    F_Hist = IP.CalcGrayHist(F_Gray)
    IP.ShowGrayHist("ForeGround Gray Hist", F_Hist)
    del IP

def MyShowColorHistogram():
    IP = cv2IP.HistIP()
    SrcImg = IP.ImRead("img/foreGroundAsset.png")
    if SrcImg.shape[2] == 4:
        F_BGR = IP.ImBGRA2BGR(SrcImg)
    else:
        F_BGR = np.array(SrcImg)
    F_Hist = IP.CalcColorHist(F_BGR)
    IP.ImWindow("ForeGround Color Image")
    IP.ImShow("ForeGround Color Image", F_BGR)
    IP.ShowColorHist("ForeGround Color Hist", F_Hist)
    del IP

def MyMonoHistEqualize():
    IP = cv2IP.HistIP()
    SrcImg = IP.ImRead("img/InputIm_1_FixdPoint.bmp")
    if SrcImg.shape[2] == 4:
        F_BGR = IP.ImBGRA2BGR(SrcImg)
    else:
        F_BGR = np.array(SrcImg)
    F_Gray = IP.ImBGR2Gray(F_BGR)
    F_Eq = IP.MonoEqualize(F_Gray)
    F_GrayHist = IP.CalcGrayHist(F_Gray)
    F_EqualizedHist = IP.CalcGrayHist(F_Eq)
    IP.ShowGrayHist("ForeGround Gray Hist", F_GrayHist)
    IP.ShowGrayHist("ForeGround Equalized Hist", F_EqualizedHist)
    IP.ImWindow("ForeGround Gray")
    IP.ImShow("ForeGround Gray", F_Gray)
    IP.ImWindow("ForeGround Gray Equalized")
    IP.ImShow("ForeGround Gray Equalized", F_Eq)
    cv2.waitKey(0)
    del IP

def MyColorHistEqualize(CType):
    IP = cv2IP.HistIP()
    SrcImg = IP.ImRead("img/InputIm_1_FixdPoint.bmp")
    if SrcImg.shape[2] == 4:
        F_BGR = IP.ImBGRA2BGR(SrcImg)
    else:
        F_BGR = np.array(SrcImg)
    F_Eq = IP.ColorEqualize(SrcImg, CType)
    F_ColorHist = IP.CalcColorHist(SrcImg)
    F_EqualizedHist = IP.CalcColorHist(F_Eq)
    IP.ShowColorHist("ForeGround Color Hist", F_ColorHist)
    IP.ShowColorHist("ForeGround Equalized Hist", F_EqualizedHist)
    IP.ImWindow("ForeGround Color")
    IP.ImShow("ForeGround Color", SrcImg)
    IP.ImWindow("ForeGround Color Equalized")
    IP.ImShow("ForeGround Color Equalized", F_Eq)
    cv2.waitKey(0)
    del IP

def MyColorHistMatching(Epsilon):
    IP = cv2IP.HistIP()
    SrcImg = IP.ImRead("img/swan.png")
    RefImg = IP.ImRead("img/InputIm_1_FixdPoint.bmp")
    if SrcImg.shape[2] == 4:
        Src_BGR = IP.ImBGRA2BGR(SrcImg)
    else:
        Src_BGR = np.array(SrcImg)
    if RefImg.shape[2] == 4:
        Ref_BGR = IP.ImBGRA2BGR(RefImg)
    else:
        Ref_BGR = np.array(RefImg)
    OutImg = IP.HistMatching(Src_BGR, Ref_BGR, Epsilon, IP.ColorType.USE_HSV)
    IP.ImShow("Original Image", SrcImg)
    IP.ImShow("Reference Image", RefImg)
    IP.ImShow("Processed Image", OutImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    del IP


if __name__ == '__main__':
    MyColorHistMatching(0.05)
    # global GlobalEpsilon
    # GlobalEpsilon = 0.0

    # # 建立主視窗和 Frame（把元件變成群組的容器）
    # window = tk.Tk()
    # top_frame = tk.Frame(window)
    # window.title('Hist Matching')
    # window.geometry('320x140')
    # window.configure(background='white')
    # top_frame.pack()
    # header_label = tk.Label(window, text='Hist Matching')
    # header_label.pack()

    # # 以下為 ChangeEpsilon_frame 群組
    # ChangeEpsilon_frame = tk.Frame(window)
    # ChangeEpsilon_frame.pack(side=tk.TOP)

    # # 建立事件處理函式（event handler），透過元件 command 參數存取
    # def ChangeEpsilon(epsilon):
    #     header_label.config(text='Epsilon of Hist Matching is ' + epsilon)
    #     global GlobalEpsilon
    #     GlobalEpsilon = float(epsilon)

    # # 將元件分為Scale 加入主視窗
    # scale = tk.Scale(ChangeEpsilon_frame, label='epsilon', from_=0.0, to=0.1, orient=tk.HORIZONTAL,length=200, showvalue=1, tickinterval=2, resolution=0.01, command=ChangeEpsilon)
    # scale.pack()

    # # 建立事件處理函式（event handler），透過元件 command 參數存取
    # def HistMatching():
    #     MyColorHistMatching(GlobalEpsilon)

    # # 將元件分為Scale 加入主視窗
    # bottom_frame = tk.Frame(window)
    # bottom_frame.pack(side=tk.BOTTOM)
    # # 以下為 bottom 群組
    # bottom_button = tk.Button(bottom_frame, text='Generate', fg='black', command=HistMatching)
    # # 讓系統自動擺放元件（靠下方）
    # bottom_button.pack()
    
    # # 運行主程式
    # window.mainloop()
