import numpy as np
import pandas as pd
import tensorflow as tf
import random as rd
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import np_utils

#-------------------------------initial-------------------------------
flower = {}
flower_file = {}
kind = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
kind_label = {kind_name:i for i,kind_name in enumerate(kind)}
X_train = []
Y_train = []
X_test = []
Y_test = []
train_name = []
test_name = []
IMG_SIZE = (256, 256)

#---read image---
def read_flower_directory(directory_name):
    flower[kind_label[directory_name]] = []
    flower_file[kind_label[directory_name]] = []
    for filename in os.listdir('./flowers/' + directory_name):
        img = cv2.imread('flowers/' + directory_name + '/' + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        flower[kind_label[directory_name]].append(img)
        flower_file[kind_label[directory_name]].append(filename)

#---分割資料集 & shuffle---
def split_shuffle(x, y):
    shuffle = list(zip(x, y))
    np.random.shuffle(shuffle)
    x, y = zip(*shuffle)
    x = np.array(x, dtype = 'float32')
    y = np.array(y)
    return (x, y)

def flower_split_train(dct,dct_name):
    for key in dct.keys():
        dct[key], dct_name[key] = split_shuffle(dct[key], dct_name[key])
        for i in range(len(dct[key])):
            if i < len(dct[key])*0.7:
                X_train.append(dct[key][i])
                Y_train.append(key)
                train_name.append(flower_file[key][i])
            else:
                X_test.append(dct[key][i])
                Y_test.append(key)
                test_name.append(flower_file[key][i])

def shuffle_flower(x, y, z):
    shuffle = list(zip(x, y, z))
    np.random.shuffle(shuffle)
    x, y, z = zip(*shuffle)
    x = np.array(x, dtype = 'float32')
    y = np.array(y, dtype = 'int32')
    z = np.array(z)
    return (x, y, z)

#---------------------程式開始---------------------
#------讀檔------
read_flower_directory('daisy')
read_flower_directory('dandelion')
read_flower_directory('rose')
read_flower_directory('sunflower')
read_flower_directory('tulip')
#------分割------
flower_split_train(flower, flower_file)
#------shuffle---
X_train, Y_train, train_name = shuffle_flower(X_train, Y_train, train_name)
X_test, Y_test, test_name = shuffle_flower(X_test, Y_test, test_name)
#---normalization---
X_train_nor = X_train / 255
X_test_nor = X_test / 255

#---模型建立---
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=3, input_shape=(256, 256,3), padding='same',activation='relu', strides=2))
model.add(Conv2D(filters=64,kernel_size=3, input_shape=(256, 256,3), padding='same',activation='relu', strides=2))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128,kernel_size=3, input_shape=(256, 256,3), padding='same', activation='relu', strides=2))
model.add(Conv2D(filters=128,kernel_size=3, input_shape=(256, 256,3), padding='same', activation='relu', strides=2))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

#---訓練模型---
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train_nor,Y_train,batch_size=128,epochs=20)
#---預測＆結果---
pred = model.predict(X_test)
classes_x = np.argmax(pred, axis=1)
loss, accuracy = model.evaluate(X_test_nor, Y_test)
matrix = model.metrics()

#---輸出結果---
print('Test:')
print('Loss: %s\nAccuracy: %s' % (loss, accuracy))

sample = rd.sample(range(0, len(Y_test)), 10)
i = 1
for sam in sample:
    print('第', i, '取樣：檔名 ', test_name[sam] ,'預測為', kind[classes_x[sam]],'正確解為', kind[Y_test[sam]])
    i = i+1

