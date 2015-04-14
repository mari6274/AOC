__author__ = 'mario'
import cv2

from globals import MyGlobals


def skeleton_thinning():
    img = cv2.cvtColor(MyGlobals.img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("aaa", img)
    return MyGlobals.img