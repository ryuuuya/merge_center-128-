import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob


def main():
    master_path = './cut2_repair'
    class_path_list = glob.glob(master_path+"/*")
    n = 0


    for class_path in class_path_list:

        img_path_list = sorted(glob.glob(class_path+'/*'))

        for img_path in img_path_list:
            root,ext = os.path.splitext(img_path)

            if ext == '.png' or '.jpeg' or '.jpg':

                print(img_path)

            print(n)
    

            n+=1





if __name__ == '__main__':
    main()
