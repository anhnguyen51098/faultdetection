import numpy as np
import cv2
import math
from scipy import ndimage
from numpy.linalg import norm


# the rotate img function will rotate the image
# adapt with the horizontal line

def rotate_img(image_path):
    # read, crop 3 pixel each size to avoid border
    img = cv2.imread(image_path)
    size = img.shape
    img_crop = img[3:size[0] - 3, 3:size[1] - 3]
    img_save = img_crop.copy()

    # use houghline to detect line
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=80, maxLineGap=5)

    # in case that HoughLinesP cannot detect
    # we reduce minimum length of detected line
    # to increase detection posibility
    if lines is None:
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=40, maxLineGap=5)
        if lines is None:
            # change brightness of image to see line clearly
            img_crop = change_brightness(img_crop, value=50)
            img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
            lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=80, maxLineGap=5)

    angles = []
    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img_crop, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # print angle of image
    median_angle = np.median(angles)
    print(f"Angle: {median_angle:.04f}")

    if median_angle < 0:
        img_rotated = ndimage.rotate(img_save, median_angle, cval=256)
    else:
        img_rotated = ndimage.rotate(img_save, median_angle + 180, cval=256)

    # cv2.imshow('Rotation', img_rotated)
    # key = cv2.waitKey(0)
    cv2.imwrite('rotated_obj.jpg', img_rotated)

    return img_rotated


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# this function will increase or decrease the brightness into the middle value
# this function will take the norm of brightness value of pixel
# then adding the value to brighness value to 127
# *note: the brighness value is from 0 -> 255, middle point 127
def brightness(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    print(v.shape)
    small = []
    for x in range(v.shape[0]):
        for y in range(v.shape[1]):
            if v[x][y] < 200:
                small.append(v[x][y])
    average_light = np.average(small)
    print(average_light)

    if average_light < 130:
        value = 130 - average_light
    else:
        value = 0

    img_bri = change_brightness(img, value)
    return img_bri


def add_white_bg(img2):
    _, w, _ = img2.shape
    # back ground color
    color = (255, 255, 255)
    if w < 500:
        img1 = np.full((500, 500, 3), color, np.uint8)
    else:
        img1 = np.full((500, w + 150, 3), color, np.uint8)

    x_offset = y_offset = 50
    x_end = x_offset + img2.shape[1]
    y_end = y_offset + img2.shape[0]
    img1[y_offset:y_end, x_offset:x_end] = img2

    return img1
