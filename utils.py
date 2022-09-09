import cv2
import numpy as np


def threshold(img_path):
    img = cv2.imread(img_path, 0)
    thr_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, 2)
    _, thr_torezo = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)

    return thr_mean, thr_torezo

def white_bubble(img):
    return 0


def crack_detect(img):

    return 0
