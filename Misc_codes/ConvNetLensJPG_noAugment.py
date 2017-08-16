#!/usr/bin/python
'''

Author: Nesar Ramachandra 

branched from ConvNetLens_grayscaleJPG.py

Adapted from cifar10_cnn.py



'''

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras import backend as K
K.set_image_dim_ordering('tf')
import time
time_i = time.time()


num_epoch = 200
batch_size = 32
learning_rate = 0.0001
decay_rate = 0.01
opti_id = 0  # [SGD, Adam, RMSprop]
loss_id = 0


img_rows = 45
img_cols = 45
num_channel = 1
num_classes = 2
num_files = 8000*num_classes


Dir0 = '../'
Dir1 = Dir0 + 'AllTrainTestSets/JPG/'
Dir2 = ['single/', 'stack/'][1]
Dir3 = ['0/', '1/'][1]
data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
names = ['lensed', 'unlensed']
data_dir_list = ['lensed_outputs', 'unlensed_outputs']



img_data_list=[]
labels = []

# for name in names:
for labelID in [0, 1]:
	name = names[labelID]
	for img_ind in range( int(num_files/num_classes)):

		input_img =  np.load(data_path + '/'+ name +'_outputs/'+ name  + str(img_ind) + '.npy')
		if np.isnan(input_img).any(): print(labelID, img_ind, ' -- ERROR: NaN')
		else:

			img_data_list.append(input_img)
			labels.append(labelID)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')

img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
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


input_shape=img_data[0].shape

model = Sequential()

model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])



sgd = SGD(lr= learning_rate, decay= decay_rate)
adam = adam(lr= learning_rate, decay= decay_rate)
rmsprop = RMSprop(lr= learning_rate, decay= decay_rate)

opti = [sgd, adam, rmsprop][opti_id]
loss_fn = ['categorical_crossentropy'][loss_id]

#model.compile(loss=loss_fn , optimizer=opti, metrics=["accuracy"])



# opti = [sgd, adam, rmsprop][opti_id]
# loss_fn = ['categorical_crossentropy'][loss_id]

if opti_id == 0:
	sgd = SGD(lr=learning_rate, decay=decay_rate)
	# lr = 0.01, momentum = 0., decay = 0., nesterov = False
	model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
elif opti_id == 1:
	# adam = adam(lr=learning_rate, decay=decay_rate)
	adam = adam()
	# lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
	model.compile(loss= 'categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
else:
	# rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
	rmsprop = RMSprop()
	# lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
	model.compile(loss= 'categorical_crossentropy', optimizer=rmsprop, metrics=["accuracy"])




# ModelFit = model.fit(X_train, y_train, batch_size= batch_size, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
ModelFit = model.fit(X_train, y_train, batch_size= batch_size, nb_epoch= num_epoch,verbose=2, validation_split=0.2)


#--------------------------------------------------------------------------------------------------
fileOut = 'opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

model.save('../ModelOutputs/ConvLensJPG_stack_'+fileOut+'.hdf5')

#print model.summary()

epochs = np.arange(1, num_epoch+1)
train_loss = ModelFit.history['loss']
val_loss = ModelFit.history['val_loss']
train_acc = ModelFit.history['acc']
val_acc = ModelFit.history['val_acc']

training_hist = np.vstack([epochs, train_loss, val_loss, train_acc, val_acc])


np.save('../ModelOutputs/ConvLensJPG_stack_history_'+fileOut+'.npy', training_hist)


print(training_hist)


print('final acc - train and val')
print(train_acc[-1], val_acc[-1])


time_j = time.time()
print(time_j - time_i, 'seconds')
