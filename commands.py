__author__ = 'mario'
import numpy

import cv2

from globals import MyGlobals
import transformations.filters as tf
import transformations.shapes as ts

def gauss(ksizex, ksizey):
    MyGlobals.img = tf.gauss_filter(MyGlobals.img, ksizex, ksizey)

def sobel(dx, dy):
    MyGlobals.img = tf.sobel_filter(MyGlobals.img, dx, dy)


def rotate_left():
    MyGlobals.img = ts.rotate(MyGlobals.img, 90)


def rotate_right():
    MyGlobals.img = ts.rotate(MyGlobals.img, -90)


def flip_x():
    MyGlobals.img = ts.flip(MyGlobals.img, ts.X_AXIS)


def flip_y():
    MyGlobals.img = ts.flip(MyGlobals.img, ts.Y_AXIS)


def get_morphologic_kernel(kernel_type, scale):
    if kernel_type is 2:
        kernel = numpy.array([
            [0,0,1,0,0],
            [0,0,1,0,0],
            [1,1,1,1,1],
            [0,0,1,0,0],
            [0,0,1,0,0]
        ], numpy.uint8)
    elif kernel_type is 3:
        kernel = numpy.array([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ], numpy.uint8)
    elif kernel_type is 4:
        kernel = numpy.array([
            [0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]
        ], numpy.uint8)

    else:
        kernel = numpy.ones((5,5),numpy.uint8)

    b = numpy.ones((scale,scale), numpy.uint8)
    kernel = numpy.kron(kernel, b)
    return kernel


def morphologic_gradient(kernel_type = 1, scale = 1):
    kernel = get_morphologic_kernel(int(kernel_type), int(scale))
    MyGlobals.img = cv2.morphologyEx(MyGlobals.img, cv2.MORPH_GRADIENT, kernel)

def morphologic_blur(kernel_type = 1, scale = 1):
    kernel = get_morphologic_kernel(int(kernel_type), int(scale))
    a = cv2.morphologyEx(MyGlobals.img, cv2.MORPH_OPEN, kernel)
    b = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
    MyGlobals.img = b


def dft():
    m, n, depth = MyGlobals.img.shape
    m2 = cv2.getOptimalDFTSize(m)
    n2 = cv2.getOptimalDFTSize(n)
    padded = cv2.copyMakeBorder(MyGlobals.img, 0, m2-m, 0, n2-n, cv2.BORDER_CONSTANT, value=[0,0,0,0] )
    