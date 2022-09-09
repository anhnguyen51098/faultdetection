import cv2
import matplotlib.pyplot as plt
import numpy as np
import pixellib
from pixellib.instance import custom_segmentation
import tensorflow as tf

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=3, class_names=["BG", "obj", "shapenotok"])


#input the image by upload through cv2

        
def sobel_edge_detection(image_path, blur_ksize=1, sobel_ksize=1, skipping_threshold=20):
    """
    image_path: link to image
    blur_ksize: kernel size parameter for Gaussian Blurry
    sobel_ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    skipping_threshold: ignore weakly edge
    """
    # read image
    img = cv2.imread(image_path)

    # convert BGR to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # sobel algorthm use cv2.CV_64F
    sobelx64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    img_sobelx = np.uint8(abs_sobel64f)

    sobely64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
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

def check_dimension(image_path, use_dl=False, is_st=False, threshold=20):

    if use_dl == False:
        img_sobel = sobel_edge_detection(image_path, skipping_threshold=threshold)

        right_list_y = []
        top_list_x = []
        bottom_list_x = []
        left_list_y = []

        # Additional array support draw function
        right_list = []

        gray_x = img_sobel.astype(np.uint8)
        mask = np.zeros(img_sobel.shape, np.uint8)
        ret, thresh = cv2.threshold(gray_x, 200, 255, 0)
        contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 20 < cv2.contourArea(cnt):
                left = tuple(cnt[cnt[:, :, 0].argmin()][0])
                right = tuple(cnt[cnt[:, :, 0].argmax()][0])
                top = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

                right_list_y.append(right[0])
                top_list_x.append(top[1])
                bottom_list_x.append(bottom[1])
                left_list_y.append(left[0])
                right_list.append(right)

        D = max(bottom_list_x) - min(top_list_x) + 6
        B = max(right_list_y) - min(right_list_y) + 1
        A = left_list_y[1] - min(left_list_y) + 2
        W = B - A

        if is_st == True:
            img = cv2.imread(image_path)
            drawn_img = img.copy()
            # draw horizontal line
            cv2.line(drawn_img, (min(right_list)[0], 0 + 80), (min(right_list)[0], img.shape[0] - 80), (0, 0, 255), 2)
            cv2.line(drawn_img, (min(right_list)[0] + A, 0 + 80), (min(right_list)[0] + A, img.shape[0] - 80), (0, 0, 255),2)
            cv2.line(drawn_img, (max(right_list)[0], 0 + 80), (max(right_list)[0], img.shape[0] - 80), (0, 0, 255), 2)
            # draw vertical line
            cv2.line(drawn_img, (0 + 500, max(bottom_list_x) + 3), (img.shape[1] - 30, max(bottom_list_x) + 3), (0, 0, 255), 2)
            cv2.line(drawn_img, (0 + 500, min(top_list_x)), (img.shape[1] - 30, min(top_list_x)), (0, 0, 255), 2)

            # draw the measurement in order -> D -> B -> W -> A
            cv2.putText(drawn_img, str(D), (max(right_list)[0], min(top_list_x) - 3 + 30), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)

            cv2.line(drawn_img, (min(right_list)[0], max(bottom_list_x) + 3 + 150),
                    (max(right_list)[0], max(bottom_list_x)  + 3 + 150), (0, 0, 0), 2)
            cv2.putText(drawn_img, str(B), (min(right_list)[0] + A, max(bottom_list_x) + 3 + 150), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 2)

            cv2.line(drawn_img, (min(right_list)[0] + A, max(bottom_list_x) + 3 + 100),
                    (max(right_list)[0], max(bottom_list_x) + 3 + 100), (0, 0, 0), 2)
            cv2.putText(drawn_img, str(W), (min(right_list)[0] + A, max(bottom_list_x) + 3 + 100), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 2)

            cv2.line(drawn_img, (min(right_list)[0], max(bottom_list_x) + 3 + 60),
                    (min(right_list)[0] + A, max(bottom_list_x) + 3 + 60), (0, 0, 0), 2)
            cv2.putText(drawn_img, str(A), (min(right_list)[0], max(bottom_list_x) + 3 + 60), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 2)

            cv2.imwrite('drawn.png', drawn_img)


        return D, W, A, B

    else: 
        segment_image.load_model("mask_rcnn_model.002-0.142644.h5")
        segmask,_ = segment_image.segmentImage(image_path, show_bboxes=True,
                                                     extract_segmented_objects=True, save_extracted_objects=False)
        list_obj = {'obj': 1, 'cap': 2, 'capspace': 3}

        result = np.where(segmask["class_ids"] == list_obj['capspace'])
        obj = segmask["extracted_objects"]

        D = obj[result[0][0]].shape[0]
        B = obj[result[0][0]].shape[1]

        result2 = np.where(segmask["class_ids"] == list_obj['cap'])
        W = obj[result2[0][0]].shape[1]
        A = B - W

        return D, W, A, B

