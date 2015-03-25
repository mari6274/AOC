__author__ = 'mario'
import cv2


def gauss_filter(input_img, x, y):
    return cv2.GaussianBlur(input_img, (int(x), int(y)), 0)


def sobel_filter(input_img, dx, dy):
    return cv2.Sobel(input_img, -1, int(dx), int(dy), ksize=9, delta=0)