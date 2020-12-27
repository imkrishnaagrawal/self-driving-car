#%%
# Python Libs For Utility
import os
import sys
import math
import random
import ntpath
import time
from itertools import chain

#Python Mathematical Libs
import numpy as np
import pandas as pd

#Image Processing Libs
import cv2
from PIL import Image

#Machine Learning Model Libs
import tensorflow.keras as keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import MaxPooling2D,Dropout,Flatten,Dense,Conv2D

#Graphs and Plots
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ep = 5
#%%
def decode(i):    
    return np.array(list(i),dtype=np.float32)
    
def wrapper(T):
    y = []
    for i in range(T.shape[0]):
        y.append(decode(T.iloc[i].values[0]))
    return np.array(y)

def read_image(path):
    return mpimg.imread(path,cv2.IMREAD_GRAYSCALE)

def show_image(image,path=None):    
    if path == None:
        img = X[i].reshape((-1,240,320,3))
        out = np.argmax( model.predict(img))
        print(img.shape)
        img = img.reshape((240,320,3))
        plt.imshow(img,cmap=plt.cm.binary)
        plt.show()
    else:
        cv2.imshow('path',image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def removeSaturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3,1.6)
    hsv[:,:,2] = rand*hsv[:,:,2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def getROI(image):
    #return image
    polygons=np.array([ [(0,240),(0,150),(70,100),(250,100),(320,150),(320,240)]]) #[(0,240),(150,100),(300,240)]  ])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,(50, 120, 150))
    return cv2.bitwise_and(mask,image)

def detectLane(inputImage):
    inputImage = removeSaturation(inputImage)
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(inputImageGray,150,190,None, 3)
    minLineLength = 15
    maxLineGap = 7
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    if lines is not None:
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(inputImageGray,(x1,y1),(x2,y2),(0, 190, 0),5, cv2.LINE_AA)
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                cv2.polylines(inputImageGray, [pts], True,(192, 0, 142))
    return getROI(inputImage)

def getModel():
    
    model = Sequential()
    
    #Convolution Layers
    #model.add(lambda x: x/127.5-1.0,input_shape=(240,320))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(240,320, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))

    #Neural Network 
    model.add(Flatten())
    model.add(Dense(120, activation = 'elu'))
    model.add(Dense(60, activation = 'elu'))
    model.add(Dense(30, activation = 'elu'))
    model.add(Dense(3,activation='softmax'))
    lrate = 0.001
    decay = lrate/ep
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    #Compiling Model
    model.compile(optimizer=sgd,
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])
    return model

#%%
dir = '.'
df = pd.read_csv(dir+'/traning.csv',dtype={'endcodedY': str})

temp = df[['X']]
y = df[['endcodedY']]
X = np.array([np.array(detectLane(read_image(f))) for f in list(chain.from_iterable( df[['X']].values.tolist()))])
X = X.reshape((-1,240,320,3))
X = tf.keras.utils.normalize(X)
X.shape
y = wrapper(y)
X

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
model = getModel()
model.fit(X_train,y_train,epochs=ep)

#%%
val_loss, val_acc = model.evaluate(X_train, y_train)
print('Loss : ' ,val_loss)
print('Accuracy : ',val_acc)

#%%
for i in range(X_test.shape[0]):
    predictions = model.predict(X_test[i].reshape((-1,240,320,3)))
    #print(predictions)
    if np.argmax(predictions) != np.argmax(y_test[i]):
        show_image(X_test[i])
        time.sleep(2)
        print(i,end=' ')

#%%
model.save(dir+'/sdc.model')


#%%
