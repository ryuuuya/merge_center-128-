# coding: utf-8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array,array_to_img
import glob

def main():
    data_dir_path = u"./cut_out"
    file_list = os.listdir(r'./cut_out')
    data_dir_path2 = u"./cut_out2"
    file_list2 = os.listdir(r'./cut_out2')

    master_path = "./cut_out"
    master_path2 = "./cut_out2"
    class_path_list = sorted(glob.glob(master_path + "/*"))
    class_path_list2 = sorted(glob.glob(master_path2 + "/*"))

    out_list = []
    result_list = []

    if not os.path.exists("./txt_list/"):
        os.mkdir("./txt_list/")

    for class_path in class_path_list:
        #print ("class_path:",class_path)
        img_path_list = sorted(glob.glob(class_path+'/*'))
        class_name = os.path.basename(class_path)
        print("class_name:",class_name)
        #print("img_path_list:",img_path_list)
        #print(img_path_list)
        class_path2 = class_path.replace("cut_out","cut_out2")

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)
            img_path2 = img_path.replace("cut_out","cut_out2")
            #print(img_path)

            if ext == u'.png' or u'.jpeg' or u'.jpg':
                abs_name = img_path
                #abs_name = class_path + '/' + img_path
                #print("abs_name:",abs_name)
                image = cv2.imread(abs_name)
                abs_name2 = img_path2
                #abs_name2 = class_path2 + '/' + img_path
                image2 = cv2.imread(abs_name2)
                out_name = os.path.basename(img_path)
                out_name2 = os.path.basename(img_path2)
                out_list.append(out_name)
                #print("img_path:",img_path)

                after_array = np.array(image)
                before_array = np.array(image2)

                result_array = np.abs(after_array - before_array)
                sum = np.sum(result_array)
                #print(sum)
                sum = str(sum)
                result_list.append(out_name)
                result_list.append(sum)
                #result_list = '\n'.join(result_list)
                #print(out_name,':result:',result_array)
                #print(out_name,'-',out_name2,sum)
                if not os.path.exists("./txt_list/" + class_name):
                    os.mkdir("./txt_list/" + class_name)
                f = open('./txt_list/' + class_name + "/" + 'sum.txt','w')
                #f.write("\n".join(out_list))
                f.write("\n".join(result_list))
                #f.writelines(result_list)
                f.close()
        result_list = []

if __name__ == '__main__':
    main()
