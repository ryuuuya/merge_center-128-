# coding:utf-8

"""
モデルと重みの読み込み
推論を行うスクリプト

"""

import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array,array_to_img
import argparse
import os,sys
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pylab

def normalization(X):
    return X/ 127.5 -1

def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1] == 3: return X
    b = X.transpose(3,1,2,0)
    c = np.array(b[0], b[0], b[0])
    return c.transpose(3,1,2,0)

def main():

    # load model structure
    model = model_from_json(open('./saved_model/generator_model.json').read())

    # load model weights
    model.load_weights('./saved_model/generator_weights.h5', by_name=False)#64

    model.summary()

    # load image
    X = []
    #print(load_path)
    print("input image loading...")

    class_path_list = sorted(glob.glob('./cut/*'))
    #print("class_path_list:",class_path_list)
    for class_path in class_path_list:

        img_path_list = sorted(glob.glob(class_path + "/*"))
        class_path_after = class_path.replace("cut","cut_repair")
        class_path_after = class_path_after.replace("before","after")
        #print("class_path:",class_path)
        #print("img_path_list",img_path_list)
        for img_path in img_path_list:
            img = load_img(img_path, target_size=(64,64))
            imgarray = img_to_array(img)
            X.append(imgarray)

        X = np.array(X).astype(np.float32)
        X = normalization(X)

        img_X = X

        # predict
        print("let's predict")
        c = 0
        for img in img_X:
            x = []

            x.append(img)
            x = np.array(x).astype(np.float32)
            #print(x.shape)
            X_gen = model.predict(x)
            X_gen = inverse_normalization(X_gen)
            #print(X_gen.shape)
            X_gen = to3d(X_gen)
            X_res = np.concatenate(X_gen, axis=1)
            out_name = os.path.basename(img_path_list[c])
            print("X_res",X_res)
            #print(class_path_after)
            print("predict: " + class_path_after + "/" + out_name )
            X_res = cv2.cvtColor(X_res*255., cv2.COLOR_RGB2BGR)
            cv2.imwrite(class_path_after + '/' + out_name,X_res)

            X = []

            c += 1

    class_path_list2 = sorted(glob.glob('./cut_ver/*'))
    #print("class_path_list:",class_path_list)
    for class_path in class_path_list2:

        img_path_list = sorted(glob.glob(class_path + "/*"))
        class_path_after = class_path.replace("cut_ver","cut_ver_repair")
        class_path_after = class_path_after.replace("before","after")
        #print("class_path:",class_path)
        #print("img_path_list",img_path_list)
        for img_path in img_path_list:
            img = load_img(img_path, target_size=(64,64))
            imgarray = img_to_array(img)
            X.append(imgarray)

        X = np.array(X).astype(np.float32)
        X = normalization(X)

        img_X = X

        # predict
        print("let's predict")
        c = 0
        for img in img_X:
            x = []

            x.append(img)
            x = np.array(x).astype(np.float32)
            #print(x.shape)
            X_gen = model.predict(x)
            X_gen = inverse_normalization(X_gen)
            #print(X_gen.shape)
            X_gen = to3d(X_gen)
            X_res = np.concatenate(X_gen, axis=1)
            out_name = os.path.basename(img_path_list[c])
            print("X_res",X_res)
            #print(class_path_after)
            print("predict: " + class_path_after + "/" + out_name )
            X_res = cv2.cvtColor(X_res*255., cv2.COLOR_RGB2BGR)
            cv2.imwrite(class_path_after + '/' + out_name,X_res)

            X = []

            c += 1

    class_path_list2 = sorted(glob.glob('./cut_side/*'))
    #print("class_path_list:",class_path_list)
    for class_path in class_path_list2:

        img_path_list = sorted(glob.glob(class_path + "/*"))
        class_path_after = class_path.replace("cut_side","cut_side_repair")
        class_path_after = class_path_after.replace("before","after")
        #print("class_path:",class_path)
        #print("img_path_list",img_path_list)
        for img_path in img_path_list:
            img = load_img(img_path, target_size=(64,64))
            imgarray = img_to_array(img)
            X.append(imgarray)

        X = np.array(X).astype(np.float32)
        X = normalization(X)

        img_X = X

        # predict
        print("let's predict")
        c = 0
        for img in img_X:
            x = []

            x.append(img)
            x = np.array(x).astype(np.float32)
            #print(x.shape)
            X_gen = model.predict(x)
            X_gen = inverse_normalization(X_gen)
            #print(X_gen.shape)
            X_gen = to3d(X_gen)
            X_res = np.concatenate(X_gen, axis=1)
            out_name = os.path.basename(img_path_list[c])
            print("X_res",X_res)
            #print(class_path_after)
            print("predict: " + class_path_after + "/" + out_name )
            X_res = cv2.cvtColor(X_res*255., cv2.COLOR_RGB2BGR)
            cv2.imwrite(class_path_after + '/' + out_name,X_res)

            X = []

            c += 1

if __name__ == '__main__':
    main()
