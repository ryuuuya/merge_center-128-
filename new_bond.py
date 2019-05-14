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

    master_path = './cut_32'
    class_path_list = glob.glob(master_path+"/*")
    master_path2 = './cut2_32'
    class_path_list2 = glob.glob(master_path2+"/*")
    master_path_side = "./cut_side_32"
    class_path_list_side = glob.glob(master_path_side+"/*")
    master_path_ver = "./cut_ver_32"
    class_path_list_ver = glob.glob(master_path_ver+"/*")
    txt_list = []
    n=1
    #print(int(w_data[-1]))

    for class_path in class_path_list:
        class_path2 = class_path.replace("cut","cut2")
        class_path_side = class_path.replace("32","side_32")
        class_path_ver = class_path.replace("32","ver_32")
        file_name = class_path.replace(master_path,"")
        #j:横に30個結合したものを17個作る
        #i:縦に結合していく（16.375）→17回
        for i in range(int(h_data[-1]+1)):
#            print("i:",i)
            #image_h1:横ずらしの1列目 h32w64
            #image_h2:横ずらしの2列目 h32w64
            #image_s:最初に4つくっつけたの

            if  i <int(h_data[-1]):
                image_h1 = cv2.imread(class_path + "/test_" + str(i) + '_' + str(0) + '.png')
                image_h_side = cv2.imread(class_path_side + "/test_" + str(i) + '_' + str(0) + '.png')
                image_h1 = cv2.hconcat([image_h1,image_h_side])

                image_h_ver = cv2.imread(class_path_ver + "/test_" + str(i) + '_' + str(0) + '.png')
                image_h2 = cv2.imread(class_path2 + "/test_" + str(i) + '_' + str(0) + '.png')
                image_h2 = cv2.hconcat([image_h_ver,image_h2])

                image_s = cv2.vconcat([image_h1,image_h2])

                #image:cut,cut_sideの列
                #image2:cut_ver,cut2の列
                for j in range (int(w_data[-1])):
#                    print(i,j+1)
                        #image_h = cv2.hconcat([image_h,image_s3])#横一列を作る
                    if j ==21:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image_ver = cv2.imread(class_path_ver + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image_s2 = cv2.vconcat([image,image_ver])

                        image_s = cv2.hconcat([image_s,image_s2])#横一列を作る


                    else:
                        image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        #print(image.shape)
                        image_side = cv2.imread(class_path_side + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        #print(image_side.shape)
                        image = cv2.hconcat([image,image_side])

                        image_ver = cv2.imread(class_path_ver + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image1 = cv2.imread(class_path2 + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                        image1 = cv2.hconcat([image_ver,image1])

                        image_s2 = cv2.vconcat([image,image1])#続く４つくっつけたの
                        image_s = cv2.hconcat([image_s,image_s2])#横一列を作る

                img_h_list.append(image_s)
            #最後の一列
            else:
                for j in range (int(w_data[-1])):
                    #print("saigo",i,j+1)
                    if j <21:
                        image_h1 = cv2.imread(class_path + "/test_" + str(i) + '_' + str(0) + '.png')
                        image_h_side = cv2.imread(class_path_side + "/test_" + str(i) + '_' + str(0) + '.png')
                        image_s = cv2.hconcat([image_h1,image_h_side])
                        for j in range (int(w_data[-1])-1):
                            #print("kaku",i,j+1)
                            image = cv2.imread(class_path + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                            #print(image.shape)
                            image_side = cv2.imread(class_path_side + "/test_" + str(i) + '_' + str(w_data[j+1]) + '.png')
                            #print(image.shape)
                            #print(image_side.shape)
                            image_s2 = cv2.hconcat([image,image_side])
                            image_s = cv2.hconcat([image_s,image_s2])
                    else:
                            image_h1 = cv2.imread(class_path + "/test_" + str(i) + '_' + str(j+1) + '.png')
                            #print(image.shape)
                            #print(image_h1.shape)
                            image_s = cv2.hconcat([image_s,image_h1])

                img_h_list.append(image_s)
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

if __name__ == '__main__':
    main()
