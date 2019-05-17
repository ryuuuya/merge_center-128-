import cv2
import csv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os


def main():
    data_dir_path = u"./inputs"
    file_list = os.listdir(r'./inputs')

    n=1

    for file_name in file_list:
        root,ext = os.path.splitext(file_name)

        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name

            img = cv2.imread(abs_name)
            file_name_after = file_name.replace("before","after")

            if not os.path.exists("./cut/"):
                os.mkdir("./cut/")
            if not os.path.exists("./cut_repair/"_after):
                os.mkdir("./cut_repair/"_after)
            if not os.path.exists("./cut2/"):
                os.mkdir("./cut2/")
            if not os.path.exists("./cut2_repair/"_after):
                os.mkdir("./cut2_repair/"_after)
            if not os.path.exists("./cut_side/"):
                os.mkdir("./cut_side/")
            if not os.path.exists("./cut_side_repair/"_after):
                os.mkdir("./cut_side_repair/"_after)
            if not os.path.exists("./cut_ver/"):
                os.mkdir("./cut_ver/")
            if not os.path.exists("./cut_ver_repair/"):
                os.mkdir("./cut_ver_repair/")
            if not os.path.exists("./cut_64/"):
                os.mkdir("./cut_64/" +  file_name_after)
            if not os.path.exists("./cut2_64/"):
                os.mkdir("./cut2_64/" +  file_name_after)
            if not os.path.exists("./cut_side_64/"):
                os.mkdir("./cut_side_64/" +  file_name_after)
            if not os.path.exists("./cut_ver_64/"):
                os.mkdir("./cut_ver_64/" +  file_name_after)
            if not os.path.exists("./edge"):
                os.mkdir("./edge")



            if not os.path.exists("./cut/" + file_name):
                os.mkdir("./cut/" + file_name)
            if not os.path.exists("./cut_repair/" + file_name_after):
                os.mkdir("./cut_repair/" + file_name_after)
            if not os.path.exists("./cut2/" + file_name):
                os.mkdir("./cut2/" + file_name)
            if not os.path.exists("./cut2_repair/" + file_name_after):
                os.mkdir("./cut2_repair/" + file_name_after)
            if not os.path.exists("./cut_side/" + file_name):
                os.mkdir("./cut_side/" + file_name)
            if not os.path.exists("./cut_side_repair/" + file_name_after):
                os.mkdir("./cut_side_repair/" + file_name_after)
            if not os.path.exists("./cut_ver/" + file_name):
                os.mkdir("./cut_ver/" + file_name)
            if not os.path.exists("./cut_ver_repair/" + file_name_after):
                os.mkdir("./cut_ver_repair/" + file_name_after)
            if not os.path.exists("./cut_64/" + file_name_after):
                os.mkdir("./cut_64/" +  file_name_after)
            if not os.path.exists("./cut2_64/" + file_name_after):
                os.mkdir("./cut2_64/" +  file_name_after)
            if not os.path.exists("./cut_side_64/" + file_name_after):
                os.mkdir("./cut_side_64/" +  file_name_after)
            if not os.path.exists("./cut_ver_64/" + file_name_after):
                os.mkdir("./cut_ver_64/" +  file_name_after)
            if not os.path.exists("./edge"):
                os.mkdir("./edge")
            #if not os.path.exists("./parts/" + file_name):
                #os.mkdir("./parts/" + file_name)
#            os.mkdir("./分割/" + file_name)
        height, width, channels = img.shape
        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])

        img_edge = img[0:1080,0:256]
        cv2.imwrite("./edge/edge.png", img_edge)
        """
        img_above = img[0:16,256:1664]
        img_below = img[1064:1080,256:1664]
        img_edge1 = img[0:1080,0:256]
        img_edge2 = img[0:1080,1664:1920]
        cv2.imwrite("./parts/" + file_name + "/above.png", img_above)
        cv2.imwrite("./parts/" + file_name + "/below.png", img_below)
        cv2.imwrite("./parts/" + file_name + "/edge1.png", img_edge1)
        cv2.imwrite("./parts/" + file_name + "/edge2.png", img_edge2)
        """

#30,16.875,new_img_width,new_img_height:64*64にしたい
        new_img_height = 128
        new_img_width = 128
        split_data_list = []
        new_width = width - 480
        new_width2 = width - 544
        new_height2 = 1024


        height_split = int((height / new_img_height)+1) #17
        width_split = int((new_width / new_img_width)+1) #23
        height_split2 = int(new_height2 / new_img_height) #16
        width_split2 = int(new_width2 / new_img_width) #22

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
                cv2.imwrite("./cut/" + file_name + "/" + file_name2, clp)

                data_list.append(file_name2)
                data_list.append(h)
                data_list.append(w)
                split_data_list.append(data_list)

        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])
            writer.writerows(split_data_list)


        for h in range(height_split2):
            height_start = (h * new_img_height) + 32 #64*n
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
                cv2.imwrite("./cut_ver/" + file_name + "/" + file_name2, clp)

                data_list.append(file_name2)
                data_list.append(h)
                data_list.append(w)
                split_data_list.append(data_list)


        for h in range(height_split):
            if h ==16:
                height_start = height - new_img_height
                height_end = height
            else:
                height_start = h * new_img_height #64*n
                height_end = height_start + new_img_height

            for w in range(width_split2):
                data_list = []
                width_start = (w * new_img_width) + 272
                width_end = width_start + new_img_width

                file_name2 = "test_" + str(h) + "_" + str(w) + ".png"
                clp = img[height_start:height_end, width_start:width_end]
                cv2.imwrite("./cut_side/" + file_name + "/" + file_name2, clp)

                data_list.append(file_name2)
                data_list.append(h)
                data_list.append(w)
                split_data_list.append(data_list)

        for h in range(height_split2):
            height_start = (h * new_img_height) + 32 #64*n
            height_end = height_start + new_img_height

            for w in range(width_split2):
                data_list = []
                width_start = (w * new_img_width) + 272
                width_end = width_start + new_img_width

                file_name2 = "test_" + str(h) + "_" + str(w) + ".png"
                clp = img[height_start:height_end, width_start:width_end]
                cv2.imwrite("./cut2/" + file_name + "/" + file_name2, clp)

                data_list.append(file_name2)
                data_list.append(h)
                data_list.append(w)
                split_data_list.append(data_list)


if __name__ == '__main__':
    main()
