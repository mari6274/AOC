# coding=utf-8

__author__ = 'mario'

import Tkinter
import tkFileDialog
import thread
from Tkinter import W, E

from commands import *


cv2.namedWindow("PREVIEW", cv2.CV_WINDOW_AUTOSIZE)

def img_show():
    while True:
        cv2.imshow("PREVIEW", MyGlobals.img)
        cv2.waitKey(30)


def gauss_ui():
    gauss_window = Tkinter.Tk()
    gauss_window.wm_title("gauss")

    odds = list(xrange(1, 99, 2))

    Tkinter.Label(gauss_window, text="Kernel size x").pack()
    kernel_size_x_spinbox = Tkinter.Spinbox(gauss_window, values=odds)
    kernel_size_x_spinbox.pack()
    Tkinter.Label(gauss_window, text="Kernel size y").pack()
    kernel_size_y_spinbox = Tkinter.Spinbox(gauss_window, values=odds)
    kernel_size_y_spinbox.pack()
    button = Tkinter.Button(gauss_window, text="OK", command = lambda: gauss(kernel_size_x_spinbox.get(), kernel_size_y_spinbox.get()))
    button.pack()


def sobel_ui():
    sobel_window = Tkinter.Tk()
    sobel_window.wm_title("sobel")

    Tkinter.Label(sobel_window, text="dx").pack()
    dx_spinbox = Tkinter.Spinbox(sobel_window, values=list(xrange(0,100)))
    dx_spinbox.pack()
    Tkinter.Label(sobel_window, text="dy").pack()
    dy_spinbox = Tkinter.Spinbox(sobel_window, values=list(xrange(0,100)))
    dy_spinbox.pack()
    button = Tkinter.Button(sobel_window, text="OK", command = lambda: sobel(dx_spinbox.get(), dy_spinbox.get()))
    button.pack()


def morphologic_ui():
    morphologic_window = Tkinter.Tk()
    morphologic_window.wm_title("morphologic")

    kernel_type_scale = Tkinter.Scale(morphologic_window, from_=1, to=4, label="kernel type", orient=Tkinter.HORIZONTAL)
    kernel_type_scale.pack(fill=Tkinter.BOTH, side=Tkinter.TOP)
    kernel_scale_scale = Tkinter.Scale(morphologic_window, from_=1, to=20, label="kernel size", orient=Tkinter.HORIZONTAL)
    kernel_scale_scale.pack(fill=Tkinter.BOTH, side=Tkinter.TOP)

    button = Tkinter.Button(morphologic_window, text="gradient", command= lambda : morphologic_gradient(kernel_type_scale.get(), kernel_scale_scale.get()))
    button.pack(fill=Tkinter.BOTH, side=Tkinter.LEFT)

    button2 = Tkinter.Button(morphologic_window, text="blur", command= lambda : morphologic_blur(kernel_type_scale.get(), kernel_scale_scale.get()))
    button2.pack(fill=Tkinter.BOTH, side=Tkinter.RIGHT)


def load_file():
    filename = tkFileDialog.askopenfilename()
    temp = cv2.imread(filename, cv2.CV_WINDOW_AUTOSIZE)
    if temp is not None:
        MyGlobals.img = temp
        MyGlobals.backup = temp
        thread.start_new_thread(img_show, ())


def save_file():
    out_file = tkFileDialog.asksaveasfilename(defaultextension=".txt")
    cv2.imwrite(out_file, MyGlobals.img)


def clean_changes():
    MyGlobals.img = MyGlobals.backup


def ui():
    window = Tkinter.Tk()
    window.wm_title("AOC")

    loadfilebutton = Tkinter.Button(window, text="load file...", command=load_file)
    loadfilebutton.grid(row=0, column=1, sticky=W+E)
    savefilebutton = Tkinter.Button(window, text="save file as...", command=save_file)
    savefilebutton.grid(row=0, column=2, sticky=W+E)

    gaussbutton = Tkinter.Button(window, text="gauss", command=gauss_ui)
    gaussbutton.grid(row=2, column=1, sticky=W+E)
    sobelbutton = Tkinter.Button(window, text="sobel", command=sobel_ui)
    sobelbutton.grid(row=3, column=1, sticky=W+E)
    morphologicbutton = Tkinter.Button(window, text="morphologic transformations", command=morphologic_ui)
    morphologicbutton.grid(row=2, column=2, sticky=W+E)
    rotateleftbutton = Tkinter.Button(window, text="rotate left", command=rotate_left)
    rotateleftbutton.grid(row=4, column=1, sticky=W+E)
    rotaterightbutton = Tkinter.Button(window, text="rotate right", command=rotate_right)
    rotaterightbutton.grid(row=4, column=2, sticky=W+E)
    flipxbutton = Tkinter.Button(window, text="flip X axis", command=flip_x)
    flipxbutton.grid(row=5, column=1, sticky=W+E)
    flipybutton = Tkinter.Button(window, text="flip Y axis", command=flip_y)
    flipybutton.grid(row=5, column=2, sticky=W+E)
    backupbutton = Tkinter.Button(window, text="clean changes", command=clean_changes)
    backupbutton.grid(row=6, column=1, sticky=W+W)

    window.mainloop()


ui()