'''
Author: Nesar Ramachandra 

Implementation of regression using ConvNet. 
Inverse problem finds 5 parameters corresponding to each image 

Make changes to include number of parameters to be backtracked, and which ones

'''



import numpy as np

import load_train_data
from model_architectures import basic_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import backend as K
K.set_image_dim_ordering('tf')
import time
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras.preprocessing.image import ImageDataGenerator
data_augmentation = True

time_i = time.time()

Dir1 = '../../AllTrainTestSets/JPG/'
Dir2 = ['single/', 'stack/'][1]
Dir3 = ['0/', '1/'][1]
data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
names = ['lensed', 'unlensed']
data_dir_list = ['lensed_outputs', 'unlensed_outputs']

num_epoch = 300
batch_size = 256
learning_rate = 0.001  # Warning: lr and decay vary across optimizers
decay_rate = 0.1
opti_id = 1  # [SGD, Adam, Adadelta, RMSprop]
loss_id = 0 # [mse, mae] # mse is always better

image_size = img_rows = 45
img_cols = 45
num_channel = 1
num_classes = 2
num_files = 8000*num_classes
train_split = 0.8   # 80 percent
num_train = int(train_split*num_files)
num_para = 5

'''
def load_train():
    img_data_list = []
    # labels = []

    # for name in names:
    for labelID in [0, 1]:
        name = names[labelID]
        for img_ind in range( int(num_files / num_classes) ):

            input_img = np.load(data_path + '/' + name + '_outputs/' + name + str(img_ind) + '.npy')
            if np.isnan(input_img).any():
                print(labelID, img_ind, ' -- ERROR: NaN')
            else:

                img_data_list.append(input_img)
                # labels.append([labelID, 0.5*labelID, 0.33*labelID, 0.7*labelID, 5.0*labelID] )

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    # labels = np.array(labels)
    # labels = labels.astype('float32')

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
    # y_train = np_utils.to_categorical(labels, num_classes)
    labels = np.load(Dir1 + Dir2 + Dir3 + 'Train5para.npy')
    # print labels1.shape
    print(labels.shape)

    para5 = labels[:,2:]
    np.random.seed(12345)
    shuffleOrder = np.arange(X_train.shape[0])
    np.random.shuffle(shuffleOrder)
    X_train = X_train[shuffleOrder]
    y_train = para5[shuffleOrder]

    # print y_train[0:10]
    # print y_train[0:10]

    return X_train, y_train

def read_and_normalize_train_data():
    train_data, train_target = load_train()
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    m = train_data.mean()
    s = train_data.std()

    print('Train mean, sd:', m, s )
    train_data -= m
    train_data /= s
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target



train_data, train_target = read_and_normalize_train_data()

X_train = train_data[0:num_train,:,:,:]
y_train = train_target[0:num_train]

X_test = train_data[num_train:num_files,:,:,:]
y_test = train_target[num_train:num_files]
'''





##-------------------------------------------------------------------------------------
## Load data

lens = load_train_data.LensData(data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/')
(X_train, y_train), (X_test, y_test) = lens.load_data()[:100]

##-------------------------------------------------------------------------------------












model = create_model()

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

# Fit the model on the batches generated by datagen.flow().
ModelFit = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=num_epoch, verbose = 2, 
                    validation_data= (X_test, y_test ) )




plotLossAcc = False
if plotLossAcc:
    import matplotlib.pylab as plt

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    # train_acc= ModelFit.history['acc']
    # val_acc= ModelFit.history['val_acc']
    epochs= range(1, num_epoch+1)


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss)
    ax.plot(epochs,val_loss)
    ax.set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss','val_loss'])

    # accuracy doesn't make sense for regression

    plt.show()


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epoch+1)
    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])


    fileOut = 'DeeperRegressionStack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    print(fileOut)
    model.save('../../ModelOutRegression/' + fileOut + '.hdf5')
    np.save('../../ModelOutRegression/'+fileOut+'.npy', training_hist)

time_j = time.time()
print(time_j - time_i, 'seconds')
