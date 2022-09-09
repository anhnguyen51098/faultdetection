import cv2
import pixellib
from pixellib.instance import custom_segmentation
import faultdetection_m3.rotate_image_module as rt
import matplotlib.pyplot as plt


segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=1, class_names=["BG", "obj"])


def extract_image(image_path):
    rt.rotate_img(image_path)
    segment_image.load_model("mask_rcnn_model.016-0.067931.h5")
    segmask, output = segment_image.segmentImage('rotated_obj.jpg', show_bboxes=True,
                                                 output_image_name='detect_obj.jpg', extract_segmented_objects=True,
                                                 save_extracted_objects=False)

    coor = segmask["rois"]
    coor = sorted(coor, key=lambda x: x[0])
    # load image
    img = plt.imread('rotated_obj.jpg')
    count_obj = 0
    list_image = []

    for [x1, y1, x2, y2] in coor:

        img_cropped = img[x1:x2, y1:y2, :]
        ###
        img_treat = rt.brightness(img_cropped)
        img_final = rt.add_white_bg(img_treat)
        ###
        num = str(count_obj)
        cv2.imwrite('obj ' + num + '.jpg', img_final)
        list_image.append('obj ' + num + '.jpg')
        count_obj += 1

    return list_image



