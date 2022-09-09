from imp import load_module
from cv2 import threshold
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image
from dimension_module import sobel_edge_detection
import numpy as np
import cv2
import math
from scipy import ndimage
import faultdetection_m3.rotate_image_module as rt
from pixellib.instance import custom_segmentation
from numpy.linalg import norm


st.title("""XRAY AI PROJECT - Demo""")
st.subheader("Demo of XRAY AI Project")

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=2, class_names=["BG", "obj", "shapenotok"])
segment_image.load_model("mask_rcnn_model.018-0.207216.h5")


def sobel_edge_detection(gray, blur_ksize=1, sobel_ksize=1, skipping_threshold=20, v_or_h=0):
    """
    image_path: link to image
    blur_ksize: kernel size parameter for Gaussian Blurry
    sobel_ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    skipping_threshold: ignore weakly edge
    """

    # convert BGR to gray
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if v_or_h == 0:
        # sobel algorthm use cv2.CV_64F for vertical
        sobelx64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        abs_sobel64f = np.absolute(sobelx64f)
        img_sobelx = np.uint8(abs_sobel64f)

        sobely64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        abs_sobel64f = np.absolute(sobely64f)
        img_sobely = np.uint8(abs_sobel64f)
    else:
        # sobel algorthm use cv2.CV_64F for horizontal
        sobelx64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        abs_sobel64f = np.absolute(sobelx64f)
        img_sobelx = np.uint8(abs_sobel64f)

        sobely64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        abs_sobel64f = np.absolute(sobely64f)
        img_sobely = np.uint8(abs_sobel64f)

    # calculate magnitude
    img_sobel = (img_sobelx + img_sobely) / 3 * 2

    # ignore weakly pixel
    for i in range(img_sobel.shape[0]):
        for j in range(img_sobel.shape[1]):
            if img_sobel[i][j] < skipping_threshold:
                img_sobel[i][j] = 0
            else:
                img_sobel[i][j] = 255
    return img_sobel


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


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


##### NEWCODE
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

    if average_light < 120:
        value = 120 - average_light
    else:
        value = 0

    img_bri = change_brightness(img, value)
    plt.imshow(img_bri)

    return img_bri


###### Newcode
def check_crack(img):
    img_sobel = sobel_edge_detection(img, skipping_threshold=30, v_or_h=1)
    gray_x = img_sobel.astype(np.uint8)
    _, thresh = cv2.threshold(gray_x, 200, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Additional array support draw function

    top_list = []

    for cnt in contours:
        if 25 < cv2.contourArea(cnt):
            top = tuple(cnt[cnt[:, :, 1].argmin()][0])
            top_list.append(top)

    # print(top_list)
    print(len(top_list))

    if (len(top_list) % 2) == 0:
        return False
    else:
        return True

def check_white(img):
    img_sobel = sobel_edge_detection(img, skipping_threshold=30)
    gray_x = img_sobel.astype(np.uint8)
    _, thresh = cv2.threshold(gray_x, 200, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Additional array support draw function

    white_list = []
    for cnt in contours:
        if 25 < cv2.contourArea(cnt):
            #print(cv2.contourArea(cnt))
            white_list.append(cnt)
    print(len(white_list))
    if (len(white_list)) == 2:
        return False
    else:
        return True

def threshold_extract(img_path):
    img = cv2.imread(img_path, 0)
    thr_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, 2)
    _, thr_torezo = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)

    return thr_mean, thr_torezo

input_image = st.file_uploader(label='Insert image')

if input_image is not None:

    img = Image.open(input_image)
    temp_dir = "img.png"
    img = img.save(temp_dir)

    threshold = st.slider("Skipping threshold", 10, 40, 20, 1)
    st.image("img.png", caption="Standard")
    st.image(input_image, caption="Input image image")

    ## Rotate image to horizontal
    rt.rotate_img(temp_dir)
    st.image('rotated_obj.jpg', "Auto-rotated image")
    segmask, output = segment_image.segmentImage('rotated_obj.jpg', show_bboxes=True,
                                                 output_image_name='detect_obj.jpg', extract_segmented_objects=True,
                                                 save_extracted_objects=False)

    #######
    res1 = segmask["rois"]
    res2 = segmask["class_ids"]
    coor = res1.copy().tolist()
    class_id = res2.copy().tolist()
    data_object = []

    for i in range(len(class_id)):
        coor[i].append(class_id[i])
        data_object.append(coor[i])

    data_object = sorted(data_object, key=lambda x: x[0])
    count = sum(map(lambda x: x[4] == 1, data_object))

    data_object_dict = {}
    for i in range(count):
        data_object_dict[i] = []
    print(data_object_dict)

    dictnum = -1
    for i in range(len(data_object)):
        if data_object[i][4] == 1:
            dictnum += 1
            data_object_dict[dictnum].append(data_object[i])
        else:
            data_object_dict[dictnum].append(data_object[i])

    # load image
    img = plt.imread('rotated_obj.jpg')
    count_obj = 1
    list_image = []
    st.subheader("Result")

    for obj in data_object_dict:
        for coor in (data_object_dict[obj]):
            if coor[4] == 1:
                [x1, y1, x2, y2, _] = coor
                break

        # crop image
        img_cropped = img[x1:x2, y1:y2, :]

        # confirm cropped image shape
        num = str(count_obj)
        cv2.imwrite('cropobj ' + str(count_obj) + '.jpg', img_cropped)
        img_treat = brightness(img_cropped)
        img_add_bg = add_white_bg(img_treat)

        cv2.imwrite('obj ' + str(count_obj) + '.jpg', img_add_bg)

        st.subheader("Object number " + str(count_obj))
        st.image('obj ' + str(count_obj) + '.jpg', caption="Object number " + str(count_obj))

        name = 'obj ' + str(count_obj) + '.jpg'
        img_final, img_final2 = threshold_extract(name)

        ######
        for coor in (data_object_dict[obj]):

            if(sum(map(lambda x: x[4] == 2, data_object_dict[obj]))) == 0:
                st.write('Object ' + str(obj + 1) + ' have no shape defect')
            else:
                st.write('Object ' + str(obj + 1) + ' have shape defect')

        try:
            img_sobel = sobel_edge_detection(img_final, skipping_threshold=threshold)
            gray_x = img_sobel.astype(np.uint8)
            ret, thresh = cv2.threshold(gray_x, 200, 255, 0)
            contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            right_list_y = []
            top_list_x = []
            bottom_list_x = []
            left_list_y = []

            # Additional array support draw function
            right_list = []

            ####### NEWCODE CHECK BLACK ARTIFACT
            for cnt in contours:

                if 15 < cv2.contourArea(cnt):
                    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
                    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

                    # if right[0] - left[0] < 27:
                    right_list_y.append(right[0])
                    top_list_x.append(top[1])
                    bottom_list_x.append(bottom[1])
                    left_list_y.append(left[0])
                    right_list.append(right)

            right_list_y.sort()
            right_list_y.pop(0)
            right_list.sort()
            right_list.pop(0)

            if len(right_list) > 3:
                st.write('Object have black artifact in space')
                right_list_y.pop(1)
                print(right_list_y)
                right_list.pop(1)
                print(right_list)
                D = max(bottom_list_x) - min(top_list_x)
                B = max(right_list_y) - min(right_list_y)
                A = right_list_y[1] - min(right_list_y)
                A2 = left_list_y[1] - min(right_list_y)
                W = B - A
                print(A - A2)
                if (A - A2) < 6:
                    st.write('Object have no black artifact in cap')
                else:
                    st.write('Object have black artifact in cap')

            else:
                st.write('Object have no black artifact in space')
                D = max(bottom_list_x) - min(top_list_x)
                B = max(right_list_y) - min(right_list_y)
                A = right_list_y[1] - min(right_list_y)
                A2 = left_list_y[1] - min(right_list_y)
                W = B - A
                print(A - A2)
                if (A - A2) < 6:
                    st.write('Object have no black artifact in cap')
                else:
                    st.write('Object have black artifact in cap')

            #### NEWCODE CHECK CRACK
            if (check_crack(img_final)):
                st.write('Object have crack')
            else:
                st.write('Object have no crack')

            ### NEWCODE CHECK WHITE (BUBBLE)
            if (check_white(img_final2)):
                st.write('Object have white details (bubbles)')
            else:
                st.write('Object have no white details (bubbles)')

            # # Remove this in real application
            drawn_img = img_add_bg.copy()
            # # draw horizontal line
            cv2.line(drawn_img, (min(right_list)[0], 0), (min(right_list)[0], img.shape[0]), (0, 0, 255), 2)
            cv2.line(drawn_img, (min(right_list)[0] + A, 0), (min(right_list)[0] + A, img.shape[0]), (0, 0, 255), 2)
            cv2.line(drawn_img, (max(right_list)[0], 0), (max(right_list)[0], img.shape[0]), (0, 0, 255), 2)
            # draw vertical line
            cv2.line(drawn_img, (0, max(bottom_list_x) + 3), (img.shape[1], max(bottom_list_x) + 3), (0, 0, 255), 2)
            cv2.line(drawn_img, (0, min(top_list_x)), (img.shape[1], min(top_list_x)), (0, 0, 255), 2)

            # draw the measurement inorder -> D -> B -> W -> A
            cv2.putText(drawn_img, str(D), (max(right_list)[0], min(top_list_x) - 3), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)

            cv2.line(drawn_img, (min(right_list)[0], max(bottom_list_x) + 150),
                     (max(right_list)[0], max(bottom_list_x) + 150), (0, 0, 0), 2)
            cv2.putText(drawn_img, str(B), (min(right_list)[0] + A, max(bottom_list_x) + 150), cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255), 2)

            cv2.line(drawn_img, (min(right_list)[0] + A, max(bottom_list_x) + 100),
                     (max(right_list)[0], max(bottom_list_x) + 100), (0, 0, 0), 2)
            cv2.putText(drawn_img, str(W), (min(right_list)[0] + A, max(bottom_list_x) + 100), cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255), 2)

            cv2.line(drawn_img, (min(right_list)[0], max(bottom_list_x) + 50),
                     (min(right_list)[0] + A, max(bottom_list_x) + 50), (0, 0, 0), 2)
            cv2.putText(drawn_img, str(A), (min(right_list)[0], max(bottom_list_x) + 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)

            cv2.imwrite('drawn.png', drawn_img)
            st.image("drawn.png", caption="Detail Dimension")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("D Value", str(D) + " pixel")
            col2.metric("A Value", str(A) + " pixel")
            col3.metric("W Value", str(W) + " pixel")
            col4.metric("B Value", str(B) + " pixel")
            count_obj += 1
        except:
            st.subheader(
                "There may be a seriously crack defect or connection defect in the image. If image is too dark, try to lower threshold, use threshold about 16-17.")

            count_obj += 1

        # if A > W or W < 0:
        #     st.subheader("There may be a issue that not covered in this milestone in the image, try to increase threshold, use threshold 30")
