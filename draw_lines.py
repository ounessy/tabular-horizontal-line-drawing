import pytesseract
from pytesseract import Output
import cv2

import os
from shapely.geometry import Polygon
pytesseract.pytesseract.tesseract_cmd = "Tesseract path e.g  c:\Tesseract-OCR\tesseract "
import sys
from os import chdir, listdir

from os.path import join



## Hyper Params
L = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_THRESHOLD = 3
LINE_WIDTH = 2
LINE_COLOR = (0, 0, 0)



## Algo

def get_image_data(img_path):
    img = cv2.imread(img_path)
    image_to_data = pytesseract.image_to_data(img, output_type=Output.DICT)
    Xmax = img.shape[1]
    Ymax = img.shape[0]
    return image_to_data, Xmax, Ymax

def draw_lines_v1(img_path, image_to_data):
    img = cv2.imread(img_path)
    Xmax = img.shape[1]
    n_boxes = len(image_to_data['level'])
    for i in range(n_boxes):
        if filter_boxes(image_to_data, i) :
            (x, y, w, h) = (image_to_data['left'][i], image_to_data['top'][i], image_to_data['width'][i], image_to_data['height'][i])
            #cv2.line(img, (0 , y +h +5 ),(Xmax, y +h +5)  ,(0, 0, 0), 3)
            #cv2.line(img, (0 , y+h ), (Xmax + w, y + h), (0, 255, 0), 1)
            cv2.rectangle(img, (x, y), ( x + w, y + h), LINE_COLOR, LINE_WIDTH)

    """
    cv2.line(img, (0 , 0),(0, Ymax)  ,(0, 0, 0), 5)
    cv2.line(img, (0 , 0),(Xmax, 0)  ,(0, 0, 0), 5)
    cv2.line(img, (0, Ymax),(Xmax, Ymax)  ,(0, 0, 0), 5)
    cv2.line(img, (Xmax , 0),(Xmax, Ymax)  ,(0, 0, 0), 5)
    """
    cv2.namedWindow("output2", cv2.WINDOW_NORMAL)
    cv2.imshow('output2', img)



def draw_lines(img_path, image_to_data, margin = 0):
    """
    Draw extracted and filtred boxes
    """
    img = cv2.imread(img_path)
    Xmax = img.shape[1]
    Ymax = img.shape[0]

    n_boxes = len(image_to_data)
    for i in range(n_boxes-1):
        """
        For each line, we will draw a line between the bottom of the line and the next line top
        """
        (x, y, w, h) = (image_to_data[i][0], image_to_data[i][1], image_to_data[i][2], image_to_data[i][3])
        y_next = image_to_data[i+1][1]

        y_middle = (y+h+y_next)//2

        """
        To avoid the case of drawin a line over a word, we will set a threshold to y_middle, In case a hole section is not detected.
        """
        y_new = min(y_middle, y+h+margin)
        cv2.line(img, (x , y_new),(w, y_new)  ,LINE_COLOR, LINE_WIDTH)
        #cv2.line(img, (0 , y+h ), (Xmax + w, y + h), (0, 255, 0), 1)
        #cv2.rectangle(img, (x, y), ( x + w, y + h), (0, 255, 0), 1)



    cv2.line(img, (0 , 0),(0, Ymax)  ,LINE_COLOR, 5)
    cv2.line(img, (0 , 0),(Xmax, 0)  ,LINE_COLOR, 5)
    cv2.line(img, (0, Ymax),(Xmax, Ymax)  ,LINE_COLOR, 5)
    cv2.line(img, (Xmax , 0),(Xmax, Ymax)  ,LINE_COLOR, 5)

    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    #cv2.imshow('output', img)
    return img

def check_intersection(elem1, elem2):
    for l in elem1:
        if l in elem2:
            return True
    return False

## Processing extracted boxes

def check_polygon_intersection(p1, p2):
    if p1.distance(p2) == 0 :
        return True
    return False

def create_polygon(x, y, w, h):
    p = Polygon([(x, y),(x+w, y),(x+w, y + h),(x, y + h)])
    return p

def filter_boxes(image_to_data, ind):
    text = image_to_data["text"][ind]
    h = image_to_data["height"][ind]
    w = image_to_data["width"][ind]
    if len(text) > CHAR_THRESHOLD and w > h:
        return True
    return False

def process_image_to_data(image_to_data, Xmax, Ymax):

    boxes_list = list()
    boxes_list.append([0, 0, 0, 0])

    all_zero_distance = list()

    n_boxes = len(image_to_data['level'])

    """
    A first loop to merge close boxes
    """
    for i in range(n_boxes):
        if  filter_boxes(image_to_data, i) :
            (y, h) = (image_to_data['top'][i], image_to_data['height'][i])
            p1 = create_polygon(0, y, Xmax, h)

            n_b = len(boxes_list)
            flag = 0
            zero_distance = list()

            for j in range(n_b):
                elem = boxes_list[j]
                p2 = create_polygon(elem[0], elem[1], elem[2], elem[3])
                if check_polygon_intersection(p1, p2):
                    zero_distance.append(j)
                    new_y = min(y, elem[1])
                    new_h = max(y+h, elem[1] + elem[3]) - min(y, elem[1])
                    new_elem = [0, new_y, Xmax, new_h]
                    boxes_list[j]=new_elem
                    flag = 1
            if flag == 0 :
                new_elem = [0, y, Xmax, h]
                boxes_list.append(new_elem)
    return boxes_list

def clean_loop(boxes_list):

    Xmax = boxes_list[1][2]

    n = len(boxes_list)
    global_flag = 0

    all_to_be_merged = list()

    used_ind = list()

    for i in range(n):
        if i not in used_ind:
            to_be_merged = list()

            boxe1 = boxes_list[i]
            p1 = create_polygon(boxe1[0],boxe1[1],boxe1[2],boxe1[3])

            m = len(boxes_list)
            for j in range(m):
                if j not in used_ind:
                    boxe2=boxes_list[j]
                    p2 = create_polygon(boxe2[0],boxe2[1],boxe2[2],boxe2[3])
                    if check_polygon_intersection(p1, p2):
                        to_be_merged.append(boxe2)
                        used_ind.append(j)

            all_to_be_merged.append(to_be_merged)

    n_detected = len(all_to_be_merged)

    new_boxes_list = list()

    for i in range(n_detected):
        small_list = all_to_be_merged[i]
        p = len(small_list)
        new_y = min([boxe[1] for boxe in small_list])
        new_h = max([boxe[1] + boxe[3] - new_y for boxe in small_list])
        new_elem = [0, new_y, Xmax, new_h]
        new_boxes_list.append(new_elem)

    return new_boxes_list


def process_table(img_path,draw_path):
    #try:
    image_to_data, Xmax, Ymax = get_image_data(img_path)
    image_to_data = process_image_to_data(image_to_data, Xmax, Ymax)
    image_to_data = clean_loop(image_to_data)
    img = draw_lines(img_path, image_to_data, margin =2)
    image_name = os.path.basename(img_path).split(os.extsep)[0].replace(" ", "_")
    processed_im_path = draw_path+"\\"+image_name+'pro.png'

    cv2.imwrite(processed_im_path, img)


def process_path(file_path,draw_path):
    all_files = listdir(file_path)

    n = len(all_files)
    for i in range(n):
        f = all_files[i]
        img_path = join(file_path, f)
        process_table(img_path,draw_path)


