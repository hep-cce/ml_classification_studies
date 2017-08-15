#!/usr/bin/python
'''
Author: Nesar Ramachandra 

Adapted from cifar10_cnn.py

Customized for the following:
    1) 2 channels (EM and Hadronic) - every input file has shape (N,M,2)
    2) 3 labels or classes (electron, muon or tauon) - TrainingData folder should have 2 subfolders
    for each label - labels have to be manually specified though (see line 76-80)
    Note: binary2img.py shuffles and sorts .data files into TrainingData and TestData.

Issues:
    1) Gotta check kernel shapes of each convolution layer
    2) Predictions do not seem to change much for TestData files. - Try for higher epochs and
    different hyperparameters
    3) data augmentation is False right now
    4) Shows accuracy for 87% in just 2 epochs - doesn't seem right. (overfitting?)

'''

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras import backend as K
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
np.set_printoptions(precision=4)

#--------------------------------------------------------------------------------------------------


img_rows = 20
img_cols = 10
num_channel = 2
num_epoch = 100
batch_size = 32
num_classes = 4

num_files = 8800*num_classes


Dir1 = '/home/nes/Desktop/ConvNetData/atlas/4Class/'
data_path = Dir1 + 'TrainingData/'


names = ['ee', 'jet', 'mm', 'tt']

#--------------------------------------------------------------------------------------------------

# x_train, y_train, x_test, y_test = load_data(data_path)

#--------------------------------------------------------------------------------------------------

img_data_list = []
labels = []

# for name in names:
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



X_train = img_data
y_train = np_utils.to_categorical(labels, num_classes)


np.random.seed(12345)
shuffleOrder = np.arange(X_train.shape[0])
np.random.shuffle(shuffleOrder)
X_train = X_train[shuffleOrder]
y_train = y_train[shuffleOrder]


print(X_train.shape, 'train samples shape')
print(y_train.shape, 'test samples shape')

input_shape = img_data[0].shape
#~~~~~~~~~~~~~~~~~~~~~~~~~


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape= X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

sgd = SGD(lr=0.1, decay= 10**(-4.0),  momentum=0.3, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#--------------------------------------------------------------------------------------------------

ModelFit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch, verbose=1, validation_split=0.2)

# model.save('model4label.hdf5')
