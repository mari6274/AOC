__author__ = 'mario'
import cv2

X_AXIS = 0
Y_AXIS = 1
BOTH_AXIS = -1


def rotate(input_img, angle):
    width, height, depth = input_img.shape
    if angle < 0:
        transform = cv2.getRotationMatrix2D((float(width-1)/2, float(width-1)/2), angle, 1)
    else:
        transform = cv2.getRotationMatrix2D((float(height-1)/2, float(height-1)/2), angle, 1)
    return cv2.warpAffine(input_img, transform, (width, height))


def flip(input_img, axis):
    return cv2.flip(input_img, axis)