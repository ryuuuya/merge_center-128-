# coding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob
import os, sys, csv

"結合"

def main():

    h_data = []
    w_data = []
    data_list = []
    img_h_list = []

    with open('./split_data.csv','r')as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            h_data.append(int(row[1]))
            w_data.append(int(row[2]))

    master_path = './cut_repair'
    class_path_list = glob.glob(master_path+"/*")
    master_path2 = "./cut_var_repair"
    class_path_list2 = glob.glob(master_path2+"/*")
    master_path3 = "./cut_side_repair"
    class_path_list3 = glob.glob(master_path2+"/*")
    txt_list = []
    n=1

    for class_path in class_path_list:

        #data_dir_path = u"/media/futami/ボリューム/コード/ノイズフル/outputs(63018)"
        #file_list = sorted(os.listdir(r'/media/futami/ボリューム/コード/ノイズフル/outputs(63018)'))
        file_name = class_path.replace(master_path,"")
        #横に30個結合したものを17個作る
        for i in range(int(h_data[-1]+1)):
            #1〜１６まで
            if  i <int(h_data[-1]):
                image_h = cv2.imread(class_path + "/test_" + str(i) + '_' + str(0) + '.png')
                #print(image_h)
                for j in range (int(w_data[-1])):
                    if j == 21:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image = image[0:64,32:64]
                        image_h = cv2.hconcat([image_h,image])
                    else:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image_h = cv2.hconcat([image_h,image])
                img_h_list.append(image_h)
            #最後の一列
            else:
                image_h = cv2.imread(class_path + "/test_" + str(i) + '_' + str(0) + '.png')
                #image_h = image_h.crop((0,8,0,0))
                #print(image_h)
                #image_h = image_h[0:64,8:64]
                image_h = image_h[8:64,0:64]
                for j in range (int(w_data[-1])):
                    if j == 21:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        #image = image.crop((0,8,0,0))
                        image = image[8:64,32:64]
                        image_h = cv2.hconcat([image_h,image])
                    else:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        #image = image.crop((0,8,0,0))
                        image = image[8:64,0:64]
                        image_h = cv2.hconcat([image_h,image])
                img_h_list.append(image_h)

        # 縦に結合する
        build_img = img_h_list[0]
        for i in range(1, len(img_h_list)):
            #print("build_img:",build_img.shape)
            #print("img_h_list:",img_h_list[i].shape)
            # 最後のときだけ別の処理
            build_img = cv2.vconcat([build_img, img_h_list[i]])
        edge = cv2.imread("./edge/edge.png")
        build_img = cv2.hconcat([edge,build_img,edge])
        cv2.imwrite('./outputs/' + file_name,build_img)
        img_h_list =[]
        build_img = []

    for class_path in class_path_list2:

        #data_dir_path = u"/media/futami/ボリューム/コード/ノイズフル/outputs(63018)"
        #file_list = sorted(os.listdir(r'/media/futami/ボリューム/コード/ノイズフル/outputs(63018)'))
        file_name = class_path.replace(master_path2,"")
        #横に30個結合したものを17個作る
        for i in range(int(h_data[-1])):
            #1〜１5まで
            if  i <(int(h_data[-1]-1)):
                image_h = cv2.imread(class_path + "/test_" + str(i) + '_' + str(0) + '.png')
                #print(image_h)
                for j in range (int(w_data[-1]-1)):
                    if j == 20:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image = image[0:64,32:64]
                        image_h = cv2.hconcat([image_h,image])
                    else:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image_h = cv2.hconcat([image_h,image])
                img_h_list.append(image_h)
            #最後の一列
            else:
                image_h = cv2.imread(class_path + "/test_" + str(i) + '_' + str(0) + '.png')
                #image_h = image_h.crop((0,8,0,0))
                #print(image_h)
                #image_h = image_h[0:64,8:64]
                image_h = image_h[8:64,0:64]
                for j in range (int(w_data[-1]-1)):
                    if j == 20:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        #image = image.crop((0,8,0,0))
                        image = image[8:64,32:64]
                        image_h = cv2.hconcat([image_h,image])
                    else:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        #image = image.crop((0,8,0,0))
                        image = image[8:64,0:64]
                        image_h = cv2.hconcat([image_h,image])
                img_h_list.append(image_h)

        # 縦に結合する
        build_img = img_h_list[0]
        for i in range(1, len(img_h_list)):
            #print("build_img:",build_img.shape)
            #print("img_h_list:",img_h_list[i].shape)
            # 最後のときだけ別の処理
            build_img = cv2.vconcat([build_img, img_h_list[i]])
        file_name_before = file_name.replace("after","before")
        above = cv2.imread("./parts" + file_name_before + "/above.png")
        below = cv2.imread("./parts" + file_name_before + "/below.png")
        edge1 = cv2.imread("./parts" + file_name_before + "/edge1.png")
        edge2 = cv2.imread("./parts" + file_name_before + "/edge2.png")
        build_img = cv2.vconcat([above,build_img,below])
        build_img = cv2.hconcat([edge1,build_img,edge2])
        cv2.imwrite('./outputs2/' + file_name,build_img)
        img_h_list =[]
        build_img = []

if __name__ == '__main__':
    main()
