import numpy as np
from scipy.io import loadmat
import cv2
import json
import os
import pickle
import shutil
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.utils
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, Adadelta
from keras import backend as K

dimx=220
dimy=220

test_image=[]

for i in os.listdir('./test/'):
    if '.jpg' in i:
        test_image.append(i)      

my_model = keras.models.load_model('saved-model_MPIIy1.hdf5')
for i in range(len(test_image)):
    image=cv2.imread("./test/"+test_image[i],1)
    image_=image/255.0
    #image
    #plt.imshow(image)
    #plt.show()
    resize_predict=np.reshape(image_,(1,220,220,3))
    coordinates=my_model.predict(resize_predict)
    #coordinates.shape
    #coordinates
    x=coordinates[0,0:16]
    y=coordinates[0,16:]
    x=(x/2)+0.5
    y=(y/2)+0.5
    x_trans=x*dimx
    y_trans=y*dimy

    print(i)
    
    #left leg
    cv2.line(image, (int(x_trans[8]),int(y_trans[8])), (int(x_trans[9]),int(y_trans[9])), (0,255,255), 4) 
    cv2.line(image, (int(x_trans[8]),int(y_trans[8])), (int(x_trans[7]),int(y_trans[7])), (255,51,255), 4)  
    
    #right leg
    cv2.line(image, (int(x_trans[4]),int(y_trans[4])), (int(x_trans[5]),int(y_trans[5])), (255,0,0), 4)  
    cv2.line(image, (int(x_trans[5]),int(y_trans[5])), (int(x_trans[6]),int(y_trans[6])), (0,255,0), 4)  
    
    #left hand
    cv2.line(image, (int(x_trans[-1]),int(y_trans[-1])), (int(x_trans[-2]),int(y_trans[-2])), (204,102,0),4)  
    cv2.line(image, (int(x_trans[-2]),int(y_trans[-2])), (int(x_trans[-3]),int(y_trans[-3])), (0,128,255),4) 
    
    #right hand
    cv2.line(image, (int(x_trans[-4]),int(y_trans[-4])), (int(x_trans[-5]),int(y_trans[-5])), (0,204,102),4)  
    cv2.line(image, (int(x_trans[-5]),int(y_trans[-5])), (int(x_trans[-6]),int(y_trans[-6])), (204,0,204),4)
    
    #shoulder
    cv2.line(image, (int(x_trans[-4]),int(y_trans[-4])), (int(x_trans[-3]),int(y_trans[-3])), (102,0,204),4) 
    
    #hip and pelvis
    cv2.line(image, (int(x_trans[6]),int(y_trans[6])), (int(x_trans[0]),int(y_trans[0])), (0,76,153),4)  
    cv2.line(image, (int(x_trans[7]),int(y_trans[7])), (int(x_trans[0]),int(y_trans[0])), (0,255,0),4)
    
    #spine
    cv2.line(image, (int(x_trans[0]),int(y_trans[0])), (int(x_trans[1]),int(y_trans[1])), (255,255,0),4)
    
    '''
    for j in range(16):
        try:
            cv2.circle(image,(int(x[j]),int(y[j])),4,(0,0,255),-1)
            #cv2.circle(resz_img,(100,200),8,(0,255,255),-1)
        except:
            pass
    ''' 
    cv2.imwrite("./testresult/"+test_image[i], image)       
