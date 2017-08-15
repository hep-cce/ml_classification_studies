#!/usr/bin/python
'''
ConvNet implementation of classification
Runs with default parameters.

'''

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras import backend as K
K.set_image_dim_ordering('tf')
import time
from keras.preprocessing.image import ImageDataGenerator


class LensData:
    '''
        Usage - lensData().load_data

        1) input_images: loads npy files, - shuffled randomly.
        2) normalize_data: re-scaling (-1, 1)
        3) load_data: returns 2 tuples: (x_train, y_train), (x_test, y_test)
            where y_train and y_test are already one-hot-encoded (using keras.np_utils)
    '''
    def __init__(self, num_classes = 2, num_channel = 1, train_val_split = 0.8, files_per_class =
    8000):

        self.num_channel = num_channel
        self.num_classes = num_classes
        self.files_per_class = files_per_class
        self.num_files = self.files_per_class*self.num_classes
        self.train_val_split = train_val_split
        self.num_train = int(self.train_val_split*self.num_files)

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []


    def input_images(self):
        img_data_list = []
        labels = []

        file_idx = np.arange(int(self.num_files / self.num_classes))
        np.random.seed(444)
        np.random.shuffle(file_idx)

        for img_ind in file_idx:
            for labelID in [0, 1]:
                name = names[labelID]

                # print(labelID)
                input_img = np.load(data_path + '/' + name + '_outputs/' + name + str(img_ind) + '.npy')
                if np.isnan(input_img).any():
                    print (labelID, img_ind, ' -- ERROR: NaN')
                else:
                    img_data_list.append(input_img)
                    labels.append(labelID)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float32')


        labels = np.array(labels)
        labels = labels.astype('int')


        img_data /= 255.
        print (img_data.shape)

        if self.num_channel == 1:
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


        self.train_data = img_data
        # labels = np.load(Dir1 + Dir2 + Dir3 + 'Train5para.npy')
        print (labels)

        self.train_target = np_utils.to_categorical(labels, self.num_classes)

        return self.train_data, self.train_target


    def normalize_data(self):
        train_data, train_target = self.input_images()
        train_data = np.array(train_data, dtype=np.float32)
        train_target = np.array(train_target, dtype=np.float32)
        m = train_data.mean()
        s = train_data.std()

        print ('Train mean, sd:', m, s)
        train_data -= m
        train_data /= s
        print('Train shape:', train_data.shape)
        print(train_data.shape[0], 'train samples')
        return train_data, train_target


## Incorporate another shuffling - if necessary
#     def shuffle_data(self):
#         self.train_data, self.train_target = self.normalize_data()
#         np.random.seed(444)
#         shuffleOrder = np.arange(self.train_data.shape[0])
#         np.random.shuffle(shuffleOrder)
#         self.train_data = self.train_data[shuffleOrder]
#         self.train_target = self.train_target[shuffleOrder]
#
#         return self.train_data, self.train_target



    def load_data(self):
        train_data, train_target = self.normalize_data()

        self.X_train = train_data[0:self.num_train, :, :, :]
        self.y_train = train_target[0:self.num_train]

        self.X_test = train_data[self.num_train:self.num_files, :, :, :]
        self.y_test = train_target[self.num_train:self.num_files]

        return (self.X_train, self.y_train), (self.X_test, self.y_test)


def create_model_deeper(num_classes = 2, learning_rate = 0.01, decay_rate = 0.0, opti_id = 0,
                        loss_id = 0, image_size = 45):

        # input_shape=img_data[0].shape

    model = Sequential()


    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(image_size, image_size, 1) ))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3 , border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3 , border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3 , border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3 , border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same' ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    if opti_id == 0:
        sgd = SGD(lr=learning_rate, decay=decay_rate)
        # lr = 0.01, momentum = 0., decay = 0., nesterov = False
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    elif opti_id == 1:
        # Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        adam = Adam(lr = learning_rate, decay = decay_rate)
        model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=["accuracy"])

    elif opti_id == 2:
        # Adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        adadelta = Adadelta(lr = learning_rate, decay = decay_rate)
        model.compile(loss='categorical_crossentropy', optimizer= adadelta, metrics=["accuracy"])

    else:
        # rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
        rmsprop = RMSprop()
        # lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=["accuracy"])


    return model


def create_model(num_classes = 2, learning_rate = 0.01, decay_rate = 0.0, opti_id = 0,
                        loss_id = 0, image_size = 45):

	# input_shape=img_data[0].shape

	model = Sequential()

	model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=(image_size, image_size, 1)))
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
	#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])


	if opti_id == 0:
		sgd = SGD(lr=learning_rate, decay=decay_rate)
		# lr = 0.01, momentum = 0., decay = 0., nesterov = False
		model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
	elif opti_id == 1:
		adam = Adam(lr=learning_rate, decay=decay_rate)
		#adam = adam()
		# lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
		model.compile(loss= 'categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
	else:
		rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
		#rmsprop = RMSprop()
		# lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
		model.compile(loss= 'categorical_crossentropy', optimizer=rmsprop, metrics=["accuracy"])

	return model


def train(model, X_train, y_train, num_epoch = 200, batch_size = 32, train_val_split = 0.2,
          DataAugmentation = True):


    time_i = time.time()

    if DataAugmentation:

        print('Implementing pre-process and (real-time) data-augmentation (Check default options)')

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range= 0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range= 0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        ##  Compute quantities required for feature-wise normalization
        ## (std, mean, and principal components if ZCA whitening is applied).

        # image_size = np.shape(X_train[2]) # Should work for both 'tf' and 'th' ordering
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        ModelFit = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=num_epoch,
                            validation_data= (X_test, y_test ), verbose=2)

    else:
        print('No pre-processing data-augmentation')
        ModelFit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epoch,
                             verbose=2, validation_split= (1.0 - train_val_split) )
        # ModelFit = model.fit(X_train, y_train, batch_size= batch_size, nb_epoch=num_epoch, verbose=2, validation_data=(X_test, y_test))


        time_j = time.time()
        print('Training time:  ', time_j - time_i, 'seconds')

    return ModelFit


def saveModel(ModelFit, fileOut):


    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']
    train_acc = ModelFit.history['acc']
    val_acc = ModelFit.history['val_acc']

    epochs = np.arange(1, np.size(train_loss)+1)

    training_hist = np.vstack([epochs, train_loss, val_loss, train_acc, val_acc])

    # fileOut =

    model.save(fileOut+'.hdf5')
    np.save(fileOut+'.npy', training_hist)



    print('final acc - train and val')
    print(train_acc[-1], val_acc[-1])

if __name__ == "__main__":


    num_epoch = 300
    batch_size = 32
    learning_rate = 0.001  # Warning: lr and decay vary across optimizers
    decay_rate = 0.1
    opti_id = 0  # [SGD, Adam, Adadelta, RMSprop]
    loss_id = 0  # Not incorporated yet

    # image_size = 45
    # num_channel = 1
    # num_classes = 2
    # num_files = 8000*num_classes
    # train_split = 0.8   # 80 percent
    # num_train = int(train_split*num_files)


    Dir0 = '../../'
    #Dir0 = '/home/nes/Desktop/ConvNetData/lens/'
    Dir1 = Dir0 + 'AllTrainTestSets/JPG/'
    Dir2 = ['single/', 'stack/'][1]
    Dir3 = ['0/', '1/'][1]
    data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
    names = ['lensed', 'unlensed']
    data_dir_list = ['lensed_outputs', 'unlensed_outputs']


    ##-------------------------------------------------------------------------------------
    ## Load data

    lens = LensData()
    (X_train, y_train), (X_test, y_test) = lens.load_data()

    ##-------------------------------------------------------------------------------------
    ## Create model

    # model =  create_model()  # Default network
    #model = create_model_deeper()  # Deeper ConvNet
    model = create_model(learning_rate = learning_rate, decay_rate = decay_rate, opti_id = opti_id, loss_id = loss_id)   # Custom parameters

    print (model.summary())

    ##-------------------------------------------------------------------------------------
    ## Fit model

    # ModelFit = train(model, X_train, y_train)  # Train with default values
    ModelFit = train(model, X_train, y_train, num_epoch=num_epoch, batch_size=batch_size, DataAugmentation = True)  ## Train with customized parameters

    ##-------------------------------------------------------------------------------------
    ## Save model and history
    DirOut = '../../ModelOutClassification/'

    fileOut = DirOut + 'LensJPG_stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    saveModel(ModelFit, fileOut)

    #--------------------------------------------------------------------------------------------------
