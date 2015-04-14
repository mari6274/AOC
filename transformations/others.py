__author__ = 'mario'
import numpy
import cv2
from globals import MyGlobals


def get_morphological_kernel(kernel_type, scale):
    if kernel_type is 2:
        kernel = numpy.array([
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [1, 1, 1, 1, 1],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0]
                             ], numpy.uint8)
    elif kernel_type is 3:
        kernel = numpy.array([
                                 [0, 0, 1, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 1, 0, 0]
                             ], numpy.uint8)
    elif kernel_type is 4:
        kernel = numpy.array([
                                 [0, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 0]
                             ], numpy.uint8)

    else:
        kernel = numpy.ones((5, 5), numpy.uint8)

    b = numpy.ones((scale, scale), numpy.uint8)
    kernel = numpy.kron(kernel, b)
    return kernel


def morphological_gradient(kernel_type=1, scale=1):
    kernel = get_morphological_kernel(int(kernel_type), int(scale))
    return cv2.morphologyEx(MyGlobals.img, cv2.MORPH_GRADIENT, kernel)


def morphological_blur(kernel_type=1, scale=1):
    kernel = get_morphological_kernel(int(kernel_type), int(scale))
    a = cv2.morphologyEx(MyGlobals.img, cv2.MORPH_OPEN, kernel)
    b = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
    return b


def skeleton_morphological():
    ret, img = cv2.threshold(MyGlobals.img, 127, 255, cv2.THRESH_BINARY)
    skel = numpy.zeros(img.shape, numpy.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.bitwise_not(temp)
        temp = cv2.bitwise_and(img, temp)
        skel = cv2.bitwise_or(temp, skel)
        img = cv2.erode(img, element)

        if cv2.countNonZero(cv2.extractChannel(img, 0)) == 0:
            break

    return skel