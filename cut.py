# coding: utf-8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob


def main():
    master_path = './cut_repair'
    class_path_list = glob.glob(master_path+"/*")
    master_path2 = './cut2_repair'
    class_path_list2 = glob.glob(master_path2+"/*")
    master_path_side = './cut_side_repair'
    class_path_list_side = glob.glob(master_path_side+"/*")
    master_path_ver = './cut_ver_repair'
    class_path_list_ver = glob.glob(master_path_ver+"/*")
    txt_list = []
    n=0
    k=0
    s=0
    v=0

    for class_path in class_path_list:

        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)

            if ext == '.png' or '.jpeg' or '.jpg':

                if int(n+8) % 23 == 0:#一番右(x_23)の処理
                    if int(n) in range(23):#一列目の処理
                        image = cv2.imread(img_path)
                        image = image[0:48,48:64]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                    elif 161 <= int(n) <=183:
                        image = cv2.imread(img_path)
                        image = image[24:64,48:64]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[16:48,48:64]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                elif int(n+23) % 23 == 0:#一番左の処理
                    if int(n) in range(23):#一列目の処理
                        image = cv2.imread(img_path)
                        image = image[0:48,0:48]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                    elif 161 <= int(n) <=183:
                        image = cv2.imread(img_path)
                        image = image[24:64,0:48]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[16:48,0:48]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                else:
                    if int(n) in range(23):#一列目の処理
                        image = cv2.imread(img_path)
                        image = image[0:48,16:48]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                    elif 161 <= int(n) <=183:
                        image = cv2.imread(img_path)
                        image = image[24:64,16:48]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[16:48,16:48]
                        outputs_path = img_path.replace("repair","32")
                        cv2.imwrite(outputs_path,image)
                print(n)
                print(img_path)
                n+=1

                if int(n) == 391:
                    n =0

    for class_path in class_path_list2:

        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)

            if ext == '.png' or '.jpeg' or '.jpg':
                image = cv2.imread(img_path)
                image = image[16:48,16:48]
                outputs_path = img_path.replace("repair","32")
                cv2.imwrite(outputs_path,image)


    for class_path in class_path_list_side:

        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)

            if ext == '.png' or '.jpeg' or '.jpg':
                if int(s) in range(22):
                    image = cv2.imread(img_path)
                    image = image[0:48,16:48]
                    outputs_path = img_path.replace("repair","32")
                    cv2.imwrite(outputs_path,image)
                elif 154 <= int(s) <=175:
                    image = cv2.imread(img_path)
                    image = image[24:64,16:48]
                    outputs_path = img_path.replace("repair","32")
                    cv2.imwrite(outputs_path,image)
                else:
                    image = cv2.imread(img_path)
                    image = image[16:48,16:48]
                    outputs_path = img_path.replace("repair","32")
                    cv2.imwrite(outputs_path,image)
            #print(s)
            #print(img_path)
            s+=1
            if int(s) == 374:
                s =0

    for class_path in class_path_list_ver:

        #print (class_path)
        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)
            if ext == '.png' or '.jpeg' or '.jpg':

                if int(v+8) % 23 == 0:
                    image = cv2.imread(img_path)
                    image = image[16:48,48:64]
                    outputs_path = img_path.replace("repair","32")
                    cv2.imwrite(outputs_path,image)
                elif int(v+23) % 23 == 0:
                    image = cv2.imread(img_path)
                    image = image[16:48,0:48]
                    outputs_path = img_path.replace("repair","32")
                    cv2.imwrite(outputs_path,image)
                else:
                    image = cv2.imread(img_path)
                    image = image[16:48,16:48]
                    outputs_path = img_path.replace("repair","32")
                    cv2.imwrite(outputs_path,image)
            #print(v)
            #print(img_path)
            v+=1
            if int(v) == 368:
                v =0

if __name__ == '__main__':
    main()
