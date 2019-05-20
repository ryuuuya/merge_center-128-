# coding: utf-8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob

#image[height,width]
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

                if int(n+11) % 11 == 0:#一番左の処理
                    if int(n) in range(11):#一列目の処理
                        image = cv2.imread(img_path)
                        image = image[0:96,0:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,0:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                else:
                    if int(n) in range(11):#一列目の処理
                        image = cv2.imread(img_path)
                        image = image[0:96,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                print(n)
                print(img_path)
                n+=1

                if int(n) == 88:
                    n =0

    for class_path in class_path_list2:

        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)

            if ext == '.png' or '.jpeg' or '.jpg':
                if int(k+9) % 11 == 0:#一番右(x_23)の処理 #_付きの番号のため順番がおかしい
                    #一番右は横幅32だけ欲しい
                    if 77 <= int(k) <=87:#最後の一列は高さ56欲しい
                        image = cv2.imread(img_path)
                        image = image[40:128,64:128]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,64:128]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)

                else:
                    if 77 <= int(k) <=87:#最後の一列は高さ56欲しい
                        image = cv2.imread(img_path)
                        image = image[40:128,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                k+=1

                if int(k) == 88:
                    k =0

    for class_path in class_path_list_side:

        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)

            if ext == '.png' or '.jpeg' or '.jpg':
                if int(s+9) % 11 == 0:#一番右(x_23)の処理 #_付きの番号のため順番がおかしい
                    if int(s) in range(11):#一列目の処理
                        image = cv2.imread(img_path)
                        image = image[0:96,64:128]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,64:128]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                else:
                    if int(s) in range(11):#一列目の処理                    
                        image = cv2.imread(img_path)
                        image = image[0:96,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    
                #print(s)
                #print(img_path)
                s+=1
                if int(s) == 88:
                    s =0

    for class_path in class_path_list_ver:

        #print (class_path)
        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)
            if ext == '.png' or '.jpeg' or '.jpg':

                if int(v+11) % 11 == 0:#一番左の処理
                    if 77 <= int(v) <=87:#最後の一列は高さ56欲しい
                        image = cv2.imread(img_path)
                        image = image[40:128,0:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,0:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                else:
                    if 77 <= int(v) <=87:#最後の一列は高さ56欲しい
                        image = cv2.imread(img_path)
                        image = image[40:128,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
                    else:
                        image = cv2.imread(img_path)
                        image = image[32:96,32:96]
                        outputs_path = img_path.replace("repair","64")
                        cv2.imwrite(outputs_path,image)
            #print(v)
            #print(img_path)
            v+=1
            if int(v) == 88:
                v =0

if __name__ == '__main__':
    main()
