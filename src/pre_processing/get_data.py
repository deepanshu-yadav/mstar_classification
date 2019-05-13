import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import glob , pickle
import  sys
# Functions and classes for loading and using the Inception model.
#import inception
from PIL import Image
# We use Pretty Tensor to define the new classifier.
#import prettytensor as pt


def get_data(datadir, img_rows, img_cols, how_many, num_test,crop_tuple=(170,60,650,540) ):
    # datadir = args.data
    # assume each image is 512x256 split to left and right
    # print(datadir)
    imgs = glob.glob(os.path.join(datadir, '*.jpg'))


    if len(imgs)==0:
        imgs = glob.glob(os.path.join(datadir, '*.png'))
        print(len(imgs))
    # imgs = glob.glob(datadir)
    # print(len(imgs))
    data_X_train = np.zeros((how_many, img_rows, img_cols, 3))
    data_X_test = np.zeros((num_test, img_rows, img_cols, 3))
    i = 0
    for file in imgs:
        img = Image.open(file)
        # img = img.resize((img_cols, img_rows), Image.LANCZOS)
        img = img.crop(crop_tuple)
        img = img.resize((img_cols, img_rows), Image.NONE)
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        img = np.array(img)
        if i < how_many:
            data_X_train[i, :, :, :] = img
        elif i > how_many and i < how_many + num_test:
            data_X_test[i - how_many, :, :, :] = img
            # print(i)
        i = i + 1

    return data_X_train, data_X_test


def get_labels(total_len, num_classes):
    cls_train = np.zeros(total_len, dtype=int)
    labels_train = np.zeros((total_len, num_classes), dtype=int)

    parts=int( total_len/num_classes )

    for i in range(num_classes):
        labels_train[parts * i:parts * (i + 1), i] = 1
        cls_train[parts * i:parts * (i + 1)] = i

    return cls_train, labels_train


def get_labels_for_uneven_dataset(num_classes , dict_for_classes):


    total_len = 0
    for i in range(num_classes):
        total_len = total_len + dict_for_classes[i]
    cls_train = np.zeros(total_len, dtype=int)
    labels_train = np.zeros((total_len, num_classes), dtype=int)
    labels_uptill_now = 0
    for i in range(num_classes):
        parts = dict_for_classes[i]

        labels_train[labels_uptill_now:parts+labels_uptill_now, i] = 1
        cls_train[labels_uptill_now:parts+labels_uptill_now] = i
        labels_uptill_now = labels_uptill_now + parts

    return cls_train, labels_train

def get_full_path(BASE_PATH, file_name):
    return os.path.join(BASE_PATH,file_name)

def build_dataset(path , img_rows, img_cols, how_many, num_test ,if_concatenate=True,if_uneven=False):
    # for root ,dirs, files in os.walk(path ,topdown=False ):
    #    print(root ,dirs ,files)
    #glob.(os.path.join(path, '*'))
    if not if_uneven:
        g = glob.glob(os.path.join(path , '*'))
        X_train  = np.zeros(  ( len(g),  how_many, img_rows, img_cols, 3)  )
        X_test = np.zeros((len(g), num_test, img_rows, img_cols, 3))
        for i in range(len(g)):
            X_train[i] , X_test[i]    =  get_data(g[i], img_rows, img_cols, how_many, num_test  )


        if(if_concatenate):
            X_train2=np.concatenate(X_train, axis=0)
            X_test2=np.concatenate( X_test, axis=0)
            return X_train2, X_test2
        else:
            X_test = np.concatenate(X_test,axis=0)
            return X_train, X_test
    else:
        print("dd")


# def get_num_classes(path):
#     return len(glob.glob(os.path.join(path, '*')))

def get_class_names(path):
    li =glob.glob(os.path.join(path, '*'))
    to_give=[]
    for i in range(len(li)):
        name =  li[i].split('\\')
        to_give.append( name[ -1])
    return to_give




def get_single_image(path , img_rows, img_cols,crop_tuple=(170,60,650,540)):
    img = Image.open(path)
    img = img.crop(crop_tuple)
    img = img.resize((img_cols, img_rows), Image.NONE)
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')
    img = np.array(img)
    Y = np.zeros( (1,img_rows, img_cols, 3)  )

    Y[0,:,:,:] = img

    return  Y
def get_full_dataset(path , img_rows, img_cols, how_many, num_test ):

    X, Y = build_dataset(path, img_rows, img_cols, how_many, num_test)
    class_names = get_class_names(path)
    no_of_classes = len(class_names)

    cls_train, labels_train = get_labels(X.shape[0], no_of_classes )

    cls_test, labels_test = get_labels(Y.shape[0], no_of_classes)

    return X , Y, cls_train ,cls_test , labels_train , labels_test, class_names


def get_datset_for_imbalanced_classes(path,rows,cols,depth,crop_tuple=(170,60,650,540)):
    g = glob.glob(os.path.join(path, '*'))
    dict_for_classes =[]
    total= 0
    for i in range(len(g)):
        imgs = glob.glob(os.path.join(g[i], '*.jpg'))

        if len(imgs) == 0:
            imgs = glob.glob(os.path.join(g[i], '*.png'))
            #print(len(imgs))
        dict_for_classes.append( len(imgs) )
        total = total + len(imgs)

    X = np.zeros( ( total,rows,cols,depth ) )
    offset = 0
    for i in range(len(g)):
        imgs = glob.glob(os.path.join(g[i], '*.jpg'))
        if len(imgs) == 0:
            imgs = glob.glob(os.path.join(g[i], '*.png'))
        j = 0

        for file in imgs:

            img = Image.open(file)
            # img = img.resize((img_cols, img_rows), Image.LANCZOS)
            img = img.crop(crop_tuple)
            img = img.resize((rows, cols), Image.NONE)
            if img.mode != 'RGB':
                img = img.convert(mode='RGB')
            img = np.array(img)
            X[ offset + j ,:,:,: ] =img
            j = j+1
        offset = offset + dict_for_classes[i]
    cls , labels =get_labels_for_uneven_dataset(len(g) , dict_for_classes)
    class_names = get_class_names(path)
    return X , cls , labels , class_names






def get_datset_in_batch(path , img_rows, img_cols, how_many, num_test):
    X, Y = build_dataset(path, img_rows, img_cols, how_many, num_test,if_concatenate=False)
    class_names = get_class_names(path)
    no_of_classes = len(class_names)

    cls_train, labels_train = get_labels(X.shape[0]*X.shape[1], no_of_classes)

    cls_test, labels_test = get_labels(Y.shape[0]  , no_of_classes)

    return X, Y, cls_train, cls_test, labels_train, labels_test, class_names

