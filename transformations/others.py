__author__ = 'mario'
import numpy
import cv2

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


def morphological_gradient(input_img, kernel_type=1, scale=1):
    kernel = get_morphological_kernel(int(kernel_type), int(scale))
    return cv2.morphologyEx(input_img, cv2.MORPH_GRADIENT, kernel)


def morphological_blur(input_img, kernel_type=1, scale=1):
    kernel = get_morphological_kernel(int(kernel_type), int(scale))
    a = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernel)
    b = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
    return b


def skeleton_morphological(input_img):
    ret, img = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY)
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


def dft(input_img):
    dft_img = cv2.dft(numpy.float32(input_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = numpy.fft.fftshift(dft_img)
    magnitude_spectrum = 20*numpy.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return magnitude_spectrum


def segmentation(input_img, k):
    Z = input_img.reshape((-1,3))
    Z = numpy.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center=cv2.kmeans(Z,k,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = numpy.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((input_img.shape))
    return res2