#!/usr/bin/python

'''

Adapted from cifar10_cnn.py

Customized for the following:
    1) 2 channels (EM and Hadronic) - every input file has shape (N,M,2)
    2) 4 labels or classes (electron, jet, muon or tauon) - TrainingData folder should have 2 subfolders
     Note: binary2img.py shuffles and sorts .data files into TrainingData and TestData.
    3) The code reshapes (N,M,2) to (L, L, 2) for performance
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, Adamax, Nadam
from keras import backend as K
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
np.set_printoptions(precision=4)

#------------------------------------------------------------------------------

#Defining few parameters:

num_channel = 2 # Here 2 channels are EM and Hadronic calorimeters. 
num_epoch = 200
batch_size = 32
num_classes = 4 #

num_files = 8800*num_classes


Dir1 = '/...../'   # The address of the directory where the Training and Testing directories are stored
data_path = Dir1 + 'TrainingData/' # Address of the directory where the training files are stored


names = ['ee', 'jet', 'mm', 'tt'] #labels of the different objects stored in the respective directories with the same names 
#..............................................................................
# Feeding the input files in the model if the files are in .npy format :

img_data_list = []
labels = []


for labelID in [0, 1, 2, 3]:
    name = names[labelID]
    for img_ind in range(num_files / num_classes):
        fileIn = data_path + '/' + name + '/' + name  + str(img_ind) + '.npy'


        input_img = np.load(fileIn)
        if np.isnan(input_img).any():
            print labelID, img_ind, ' -- ERROR: NaN'
        else:
            img_data_list.append(input_img)
            labels.append(labelID)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = 255.*(img_data - img_data.min()) / (img_data.max() - img_data.min())
img_data /= 255
print (img_data.shape)
#..............................................................................
# Alternative way of feeding the input files if the files are in .data format :

img_data_list = []
labels = []

import glob
for labelID in [0, 1, 2]:
    name = names[labelID]
    fileIn = glob.glob("/Users/Wasikul/event_f/"+name+"/ROOT*.data")
    for file in fileIn:
        input_img = np.fromfile(file)
        input_img = input_img.reshape([60,64,2])
         if np.isnan(input_img).any():
             print labelID, img_ind, ' -- ERROR: NaN'
         else:
             img_data_list.append(input_img)
             labels.append(labelID)
img_data = np.asarray(img_data_list)
#...............................................................................

# If you use the image ordering of theano :

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print (img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print (img_data.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print (img_data.shape)

#..............................................................................
#Defining X_train and y_train:
X_train = img_data
y_train = np_utils.to_categorical(labels, num_classes)
#..............................................................................

#Shuffling to make it randomize more:
np.random.seed(12345)
shuffleOrder = np.arange(X_train.shape[0])
np.random.shuffle(shuffleOrder)
X_train = X_train[shuffleOrder]
y_train = y_train[shuffleOrder]


print(X_train.shape, 'train samples shape')
print(y_train.shape, 'test samples shape')

input_shape = img_data[0].shape

#..............................................................................
#Building the model with multiple layers

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape= X_train.shape[1:]))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

#You can make the Neural network deeper by uncommenting the follwoing:

#model.add(Conv2D(64, (3, 3), padding='same', activation='relu')
##model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(128, (3, 3), padding='same'))
#model.add(Activation('relu'))
## model.add(Conv2D(64, (3, 3)))
## model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(1, 1)))
#model.add(Dropout(0.25)

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#..............................................................................

#Define the hyperparameters :

opt = keras.optimizers.rmsprop(lr=0.0001, decay=0)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#------------------------------------------------------------------------------

# Fit and save the model:

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch = num_epoch, verbose=2, validation_split=0.2)

model.save('model_name.hdf5')

#.............................................................................

# Obtain the values of the keys of fitted model and get the values of loss and accuracy for different epochs :

print(history.history.keys())

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['acc']
val_acc=history.history['val_acc']

print(train_loss)
print(val_loss)
print(train_acc)
print(val_acc)

#.............................................................................
# Plot the loss and accuracy curves

## summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy_4.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss_4.png')
plt.show()

#.............................................................................
