#!/usr/bin/python3

import cv2
import numpy as np
import cv2IP

def MyAlPhaBlend():
    IP = cv2IP.AlphaBlend()
    
    SrcImg = IP.ImRead("img/ghost.png")
    SrcImg = cv2.resize(SrcImg, (240, 200))
    back = IP.ImRead("img/background.png")

    fore, alpha = IP.SplitAlpha(SrcImg)
    if back.shape[2] == 4: # channels: b, g, r, alpha
        back = back[:,:,:3]

    fore = np.float32(fore) # convertTo 32FC3
    alpha = np.float32(alpha) / 255.0 # convertTo 32FC3 and normalize
    back = np.float32(back)

    k = 0
    while k != 13:
        i = 0
        while i < 4 and k != 13:
            out = np.array(back)

            rows_start = 770
            rows_end = 770 + fore.shape[0]
            columns_start = 350 + 350 * i
            columns_end = 350 + fore.shape[1] + 350 * i
            
            out[rows_start:rows_end, columns_start:columns_end]= IP.MyDoBlending(fore, back[rows_start:rows_end, columns_start:columns_end], alpha, 0.25*(1+i))
            out = np.uint8(out)

            IP.ImWindow("AlphaBlending Result")
            IP.ImShow("AlphaBlending Result", out)
            k = cv2.waitKey(1000)
            i += 1

        if k != 13:
            fore_final = cv2.resize(fore, (1080, 880))
            alpha_final = cv2.resize(alpha, (1080, 880))
            out = np.array(back)
            
            rows_start = int(back.shape[0] / 2 - fore_final.shape[0] / 2)
            rows_end = int(back.shape[0] / 2 + fore_final.shape[0] / 2)
            columns_start = int(back.shape[1] / 2 - fore_final.shape[1] / 2)
            columns_end = int(back.shape[1] / 2 + fore_final.shape[1] / 2)
            
            out[rows_start:rows_end, columns_start:columns_end] = IP.MyDoBlending(fore_final, back[rows_start:rows_end, columns_start:columns_end], alpha_final, 1)
            out = np.uint8(out)

            IP.ImWindow("AlphaBlending Result")
            IP.ImShow("AlphaBlending Result", out)
            k = cv2.waitKey(1000)

    del IP


if __name__ == '__main__':
    MyAlPhaBlend()
