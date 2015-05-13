__author__ = 'mario'
import cv2
import numpy

from globals import MyGlobals
import others as to


def thinning_iteration(input_img, iter):
    marker = numpy.zeros(input_img.shape, numpy.uint8)

    rows, cols = input_img.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            p2 = input_img[i-1, j]
            p3 = input_img[i-1, j+1]
            p4 = input_img[i, j+1]
            p5 = input_img[i+1, j+1]
            p6 = input_img[i+1, j]
            p7 = input_img[i+1, j-1]
            p8 = input_img[i, j-1]
            p9 = input_img[i-1, j-1]

            a = (p2 == 0 and p3 == 1) + \
                (p3 == 0 and p4 == 1) + \
                (p4 == 0 and p5 == 1) + \
                (p5 == 0 and p6 == 1) + \
                (p6 == 0 and p7 == 1) + \
                (p7 == 0 and p8 == 1) + \
                (p8 == 0 and p9 == 1) + \
                (p9 == 0 and p2 == 1)

            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

            m1 = p2 * p4 * p6 if iter == 0 else p2 * p4 * p8
            m2 = p4 * p6 * p8 if iter == 0 else p2 * p6 * p8
            if a == 1 and (b >= 2 and b <= 6) and m1 == 0 and m2 == 0:
                marker[i,j] = 1;

    return input_img & (~marker)


def skeleton_thinning(input_img):
    input_img = to.gray_scale(input_img)
    ret, input_img = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY)
    input_img = input_img / 255

    prev = numpy.zeros(input_img.shape, numpy.uint8)
    diff = cv2.absdiff(input_img, prev)

    while cv2.countNonZero(diff) > 0:
        input_img = thinning_iteration(input_img, 0)
        input_img = thinning_iteration(input_img, 1)
        diff = cv2.absdiff(input_img, prev)
        prev = input_img

    input_img = input_img * 255
    return input_img