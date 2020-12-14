#!/usr/bin/python3

import cv2
import cv2IP

if __name__ == '__main__':
    IP = cv2IP.BaseIP()
    img = IP.ImRead("img/test.jpg")
    IP.ImWindow("foreGround")
    IP.ImShow("foreGround", img)
    cv2.waitKey(0)
    del IP
