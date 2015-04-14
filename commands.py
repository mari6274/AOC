__author__ = 'mario'

import cv2

from globals import MyGlobals
import transformations.filters as tf
import transformations.shapes as ts
import transformations.others as to
import transformations.thinning as tt


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


def morphological_gradient(kernel_type=1, scale=1):
    MyGlobals.img = to.morphological_gradient(kernel_type, scale)


def morphological_blur(kernel_type=1, scale=1):
    MyGlobals.img = to.morphological_blur(kernel_type, scale)


def dft():
    m, n, depth = MyGlobals.img.shape
    m2 = cv2.getOptimalDFTSize(m)
    n2 = cv2.getOptimalDFTSize(n)
    padded = cv2.copyMakeBorder(MyGlobals.img, 0, m2-m, 0, n2-n, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])


def skeleton_morphological():
    MyGlobals.img = to.skeleton_morphological()


def skeleton_thinning():
    MyGlobals.img = tt.skeleton_thinning()
