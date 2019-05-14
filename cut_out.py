import cv2
import csv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os


def main():
    data_dir_path = u"./outputs"
    file_list = os.listdir(r'./outputs')
    data_dir_path2 = u"./outputs2"
    file_list2 = os.listdir(r'./outputs2')

    n=1

    for file_name in file_list:
        root,ext = os.path.splitext(file_name)

        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name

            img = cv2.imread(abs_name)
            #file_name_after = file_name.replace("before","after")

            if not os.path.exists("./cut_out/" + file_name):
                os.mkdir("./cut_out/" + file_name)
            if not os.path.exists("./edge"):
                os.mkdir("./edge")

#            os.mkdir("./分割/" + file_name)
        height, width, channels = img.shape
        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])

        img_edge = img[0:1080,0:240]
        cv2.imwrite("./edge/edge.png", img_edge)

#30,16.875,new_img_width,new_img_height:64*64にしたい
        new_img_height = 64
        new_img_width = 64
        split_data_list = []
        new_width = width - 480

        height_split = int((height / new_img_height)+1) #17
        width_split = int((new_width / new_img_width)+1) #30#23

        for h in range(height_split):
            if h ==16:
                height_start = height - new_img_height
                height_end = height
            else:
                height_start = h * new_img_height #64*n
                height_end = height_start + new_img_height

            for w in range(width_split):
                data_list = []
                if w == 22:
                    width_start = 1616
                    width_end = width_start + new_img_width
                else:
                    width_start = (w * new_img_width) + 240
                    width_end = width_start + new_img_width

                file_name2 = "test_" + str(h) + "_" + str(w) + ".png"
                clp = img[height_start:height_end, width_start:width_end]
                cv2.imwrite("./cut_out/" + file_name + "/" + file_name2, clp)

                data_list.append(file_name2)
                data_list.append(h)
                data_list.append(w)
                split_data_list.append(data_list)

        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])
            writer.writerows(split_data_list)

    for file_name in file_list2:
        root,ext = os.path.splitext(file_name)

        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path2 + '/' + file_name

            img = cv2.imread(abs_name)
            #file_name_after = file_name.replace("before","after")

            if not os.path.exists("./cut_out2/" + file_name):
                os.mkdir("./cut_out2/" + file_name)
            if not os.path.exists("./edge"):
                os.mkdir("./edge")

#            os.mkdir("./分割/" + file_name)
        height, width, channels = img.shape
        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])

        img_edge = img[0:1080,0:240]
        cv2.imwrite("./edge/edge.png", img_edge)

#30,16.875,new_img_width,new_img_height:64*64にしたい
        new_img_height = 64
        new_img_width = 64
        split_data_list = []
        new_width = width - 480

        height_split = int((height / new_img_height)+1) #17
        width_split = int((new_width / new_img_width)+1) #30#23

        for h in range(height_split):
            if h ==16:
                height_start = height - new_img_height
                height_end = height
            else:
                height_start = h * new_img_height #64*n
                height_end = height_start + new_img_height

            for w in range(width_split):
                data_list = []
                if w == 22:
                    width_start = 1616
                    width_end = width_start + new_img_width
                else:
                    width_start = (w * new_img_width) + 240
                    width_end = width_start + new_img_width

                file_name2 = "test_" + str(h) + "_" + str(w) + ".png"
                clp = img[height_start:height_end, width_start:width_end]
                cv2.imwrite("./cut_out2/" + file_name + "/" + file_name2, clp)

                data_list.append(file_name2)
                data_list.append(h)
                data_list.append(w)
                split_data_list.append(data_list)

        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])
            writer.writerows(split_data_list)

if __name__ == '__main__':
    main()
