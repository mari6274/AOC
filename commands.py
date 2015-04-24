__author__ = 'mario'

import cv2

from globals import MyGlobals
import transformations.filters as tf
import transformations.shapes as ts
import transformations.others as to
import transformations.thinning as tt


def gray_scale():
    MyGlobals.img = cv2.cvtColor(MyGlobals.img, cv2.COLOR_BGR2GRAY)


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
    MyGlobals.img = to.morphological_gradient(MyGlobals.img, kernel_type, scale)


def morphological_blur(kernel_type=1, scale=1):
    MyGlobals.img = to.morphological_blur(MyGlobals.img, kernel_type, scale)


def dft():
    gray_scale()
    cv2.imshow("dft", to.dft(MyGlobals.img))

def skeleton_morphological():
    MyGlobals.img = to.skeleton_morphological(MyGlobals.img)


def skeleton_thinning():
    MyGlobals.img = tt.skeleton_thinning(MyGlobals.img)
