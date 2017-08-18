#!/usr/bin/python

'''

Adapted from mnist_autoencoder.py

Customized for the following:
1) 2 channels (EM and Hadronic) - every input file has shape (N,M,2)
2) Three classes of events are defined in this code. They are : electronelectronjetjet, muonmuonjetjet , tautaujetjet .
3) The encoder part of the code encodes (N,M,2) to (L, L, 2)  and then the decoder part of the code reconstructs the compressed representation of (L, L,2) into (N,M, 2) again.
'''



from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('tf')
np.set_printoptions(precision=4)

#.............................................................................
# Define the parameters feed the input files :

batch_size = 32
num_classes = 3
img_rows = 60
img_cols = 64
num_channel = 2
num_epoch = 20

names = ['electronelectron', 'muonmuon', 'tautau']

labels = []

import glob
zeejj = glob.glob(#provide the address of the first event directory)
zmumujj = glob.glob(#provide the address of the 2nd event directory)
zttjj = glob.glob(#provide the address of 3rd event directory)

imagelist = []
imagelist_zeejj = []
imagelist_zmumujj = []
imagelist_zttjj = []

for file in zeejj:
   image = np.fromfile(file)
   image = image.reshape([60,64,2])
   imagelist.append(image)
   
for file in zmumujj:
   image = np.fromfile(file)
   image = image.reshape([60,64,2])
   imagelist.append(image)
   
for file in zttjj:
   image = np.fromfile(file)
   image = image.reshape([60, 64,2])
   imagelist.append(image)

img_data = np.asarray(imagelist)
print(img_data.shape)

#..............................................................................
#Defne X_train and x_test :

split_ratio = 0.8
X_train = img_data[: int(split_ratio*np.shape(img_data)[0])] 
x_test = img_data[ int(split_ratio*np.shape(img_data)[0]):] 
print(X_train.shape)

#..............................................................................

#Define input shape :
input_shape = X_train[0].shape

print(input_shape)

#..............................................................................

# Encode the image :
input_img = Input(shape=input_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

encoded_shape = encoded.shape
print(encoded_shape)

#..............................................................................

#Decode the encoded image :

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)
x = ZeroPadding2D( ((0, 0), (1, 0) ) )(x)
x = ZeroPadding2D( ((0, 0), (1, 0) ) )(x)
x = ZeroPadding2D( ((0, 0), (1, 0) ) )(x)
decoded = ZeroPadding2D(((0, 0), (1, 0)))(x) 

#..............................................................................
# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)
print("autoencoder model created")

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

#..............................................................................

#Fit and save the autoencoder :

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


history = autoencoder.fit(X_train, X_train,
                          batch_size=batch_size, epochs=num_epoch,
                          verbose=1, validation_split=0.2)

print(history.history.keys())

encoder.save('encoder_name.hdf5')
autoencoder.save('autoencoder_name.hdf5')

#..............................................................................

#Get the values of train_loss and val_loss for different epochs :

print(history.history.keys())

train_loss=history.history['loss']
val_loss=history.history['val_loss']

print(train_loss)
print(val_loss)

#..............................................................................
# Geneate the loss curve for training and validation :

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss_autoencoder.png')

#..............................................................................
#produce the plots of original and decoded imapges for comparison :

decoded_imgs = autoencoder.predict(x_test)

print(decoded_imgs.shape)
print(x_test.shape)
n = 10  # how many digits we will display
fig = plt.figure(figsize=(20, 4))
for i in range(10):
    # display original image of the 1st channel
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(60,64,2)[:,:,0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstructed image of the 1st channel
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(60, 64,2)[:,:,0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
fig.savefig("autoencoded_digits.png")

#...............................................................................
