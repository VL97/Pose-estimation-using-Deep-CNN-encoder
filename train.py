import pickle
import numpy as np
import cv2
import os
import tensorflow as tf
import keras.utils
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, Adadelta
from keras import backend as K
from keras.callbacks import ModelCheckpoint


dimx=220
dimy=220
filepath = "saved-model_MPIIy1.hdf5"

#############################################
#load label dictionary
infile = open('labelsdict_mpii','rb')
labels = pickle.load(infile)
infile.close()

#############################################

X_train=[]

for i in os.listdir('./train/'):
    if '.jpg' in i:
        X_train=np.append(X_train,i)

X_valid=[]

for i in os.listdir('./valid/'):
    if '.jpg' in i:
        X_valid=np.append(X_valid,i)
        
#############################################

class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels,path_X,shuffle=False):
        #'Initialization'
        self.dim = dim                                               
        self.batch_size = batch_size                                   
        self.labels = labels                                         
        self.list_IDs = list_IDs                                     
        self.n_channels = n_channels                                 
        self.shuffle = shuffle                        
        self.path_X=path_X                            
        self.on_epoch_end()                           

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__itercustom__(list_IDs_temp)
        
        X=X/255.0   
        return X, Y

    #version tf: 1.0!!!
    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __itercustom__(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, 16*2))
            
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,:,:,:] = cv2.imread(self.path_X + ID,1)            
            Y[i,:] = np.append(labels[ID]['x'], labels[ID]['y'], axis=0)

        return X, Y
    
#############################################
        
path_to_X_train = './train/'
path_to_X_valid = './valid/'
batch_size_train=128
batch_size_valid=128

training_generator = DataGenerator(list_IDs=X_train, labels=labels, \
                                   batch_size=batch_size_train, dim=(dimy,dimx),\
                                   n_channels=3, path_X=path_to_X_train, \
                                   shuffle=True)

validation_generator = DataGenerator(list_IDs=X_valid, labels=labels,\
                                     batch_size=batch_size_valid, dim=(dimy,dimx),\
                                   n_channels=3, path_X=path_to_X_valid, \
                                   shuffle=True)

#############################################

def scheduler(epoch, lr):
  if epoch < 30:
    return 0.0005
  else:
    return 0.0005 * np.exp(0.1 * (30 - epoch))

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                             save_best_only=True, save_weights_only=False)

#Callback for lr updation and stop training 
class myCallback(keras.callbacks.LearningRateScheduler):
    def __init__(self, schedule):
        super(myCallback, self).__init__(schedule)     

    def on_epoch_end(self, epoch, logs={}):
        lr_=float(keras.backend.get_value(self.model.optimizer.lr))
        print('lr: ','{:.8f}'.format(lr_))    
    
lr_schedule = myCallback(scheduler)

#############################################

model = Sequential()


model.add(DepthwiseConv2D(kernel_size=(1,1), strides=(4, 4), depth_multiplier=1,\
                use_bias=False,input_shape=(220,220,3)))
model.add(Conv2D(96, (11, 11),activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(16*2, activation='tanh'))
    
model.compile(loss=tf.losses.mean_squared_error, optimizer=Adam())
model.summary()

#############################################

model.fit_generator(generator=training_generator,validation_data=validation_generator, \
                    epochs=80,verbose=1, callbacks=[checkpoint, lr_schedule])

#############################################