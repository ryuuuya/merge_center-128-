import cv2
import csv
import argparse
import os,sys
import glob
import pylab
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array,array_to_img
import matplotlib.pyplot as plt

def split():
    print("split")
    data_dir_path = u"./inputs"
    file_list = os.listdir(r'./inputs')

    n=1

    for file_name in file_list:
        root,ext = os.path.splitext(file_name)

        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name

            img = cv2.imread(abs_name)
            file_name_after = file_name.replace("before","after")

            if not os.path.exists("./cut"):
                os.mkdir("./cut")
            if not os.path.exists("./cut2"):
                os.mkdir("./cut2")
            if not os.path.exists("./cut2_32"):
                os.mkdir("./cut2_32")
            if not os.path.exists("./cut2_repair"):
                os.mkdir("./cut2_repair")
            if not os.path.exists("./cut_32"):
                os.mkdir("./cut_32")
            if not os.path.exists("./cut_repair"):
                os.mkdir("./cut_repair")
            if not os.path.exists("./cut_side"):
                os.mkdir("./cut_side")
            if not os.path.exists("./cut_side_32"):
                os.mkdir("./cut_side_32")
            if not os.path.exists("./cut_side_repair"):
                os.mkdir("./cut_side_repair")
            if not os.path.exists("./cut_ver"):
                os.mkdir("./cut_ver")
            if not os.path.exists("./cut_ver_32"):
                os.mkdir("./cut_ver_32")
            if not os.path.exists("./cut_ver_repair"):
                os.mkdir("./cut_ver_repair")
            if not os.path.exists("./outputs"):
                os.mkdir("./outputs")

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
            if not os.path.exists("./cut_32/" + file_name_after):
                os.mkdir("./cut_32/" +  file_name_after)
            if not os.path.exists("./cut2_32/" + file_name_after):
                os.mkdir("./cut2_32/" +  file_name_after)
            if not os.path.exists("./cut_side_32/" + file_name_after):
                os.mkdir("./cut_side_32/" +  file_name_after)
            if not os.path.exists("./cut_ver_32/" + file_name_after):
                os.mkdir("./cut_ver_32/" +  file_name_after)
            if not os.path.exists("./edge"):
                os.mkdir("./edge")
            #if not os.path.exists("./parts/" + file_name):
                #os.mkdir("./parts/" + file_name)
#            os.mkdir("./分割/" + file_name)
        height, width, channels = img.shape
        with open('./split_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'h', 'w'])

        img_edge = img[0:1080,0:240]
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
        new_img_height = 64
        new_img_width = 64
        split_data_list = []
        new_width = width - 480
        new_width2 = width - 512
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

def normalization(X):
    return X/ 127.5 -1

def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1] == 3: return X
    b = X.transpose(3,1,2,0)
    c = np.array(b[0], b[0], b[0])
    return c.transpose(3,1,2,0)

def predict():
    print("predict")

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
            #print("X_res",X_res)
            #print(class_path_after)
            #print("predict: " + class_path_after + "/" + out_name )
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
            #print("X_res",X_res)
            #print(class_path_after)
            #print("predict: " + class_path_after + "/" + out_name )
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
            #print("X_res",X_res)
            #print(class_path_after)
            #print("predict: " + class_path_after + "/" + out_name )
            X_res = cv2.cvtColor(X_res*255., cv2.COLOR_RGB2BGR)
            cv2.imwrite(class_path_after + '/' + out_name,X_res)

            X = []

            c += 1

    class_path_list2 = sorted(glob.glob('./cut2/*'))
    #print("class_path_list:",class_path_list)
    for class_path in class_path_list2:

        img_path_list = sorted(glob.glob(class_path + "/*"))
        class_path_after = class_path.replace("cut2","cut2_repair")
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
            #print("X_res",X_res)
            #print(class_path_after)
            #print("predict: " + class_path_after + "/" + out_name )
            X_res = cv2.cvtColor(X_res*255., cv2.COLOR_RGB2BGR)
            cv2.imwrite(class_path_after + '/' + out_name,X_res)

            X = []

            c += 1

def cut():
    print("cut")
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
                #print(n)
                #print(img_path)
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

def bond():
    print("bond")

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

def main():
    split()
    predict()
    cut()
    bond()

if __name__ == '__main__':
    main()
