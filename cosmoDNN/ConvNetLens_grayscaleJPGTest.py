#!/usr/bin/python
'''
branched from ConvNetLens_grayscaleWorks.py

Adapted from cifar10_cnn.py

Customized for the following:
    1) 1 channels (grayscale) - every input file has shape (M,M,2)
    2) 2 labels or classes (Lensed or Unlensed) - TrainingData folder should have 2 subfolders
    for each label - labels have to be manually specified though.
    Note: fits2img.py shuffles and sorts .data files into TrainingData and TestData.
    3) The code reshapes (M,M,2) to (L, L, 2) for performance

Issues:
    1) Gotta check kernel shapes of each convolution layer
    2) Predictions do not seem to change much for TestData files. - Try for higher epochs and
    different hyperparameters
    3) data augmentation is False right now
    4) Shows accuracy for 87% in just 2 epochs - doesn't seem right. (overfitting?)



'''

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras import backend as K
import glob
K.set_image_dim_ordering('tf')
from keras.models import load_model
import time
time_i = time.time()



Dir1 = '/home/nes/Desktop/ConvNetData/lens/AllTrainTestSets/JPG/'
Dir2 = ['single/', 'stack/'][1]
Dir3 = ['0/', '1/'][1]
data_path = Dir1 + Dir2 + Dir3 + 'TestData/'
names = ['lensed', 'unlensed']
data_dir_list = ['lensed_outputs', 'unlensed_outputs']

img_rows = 45
img_cols = 45
num_channel = 1
num_epoch = 200
batch_size = 32

num_classes = 2
num_files = 2000*num_classes


#--------------------------------------------------------------------------------------------------

# model.save('ModelOutputs/Stackmodel_200runsJPG_Test.hdf5')
# loaded_model=load_model('ModelOutputs/Stackmodel_200runsJPG_Test.hdf5')


# Or load model from hdf5 instead of json
# loaded_model=load_model('ModelOutputs/jupiter/ModelOutClassification'
# 				  '/DeeperLensJPG_stack_opti0_loss0_lr0.0001_decay0.1_batch32_epoch200.hdf5')



DirIn = '/home/nes/Dropbox/Argonne/lensData/ModelOutputs/jupiter/ModelOutClassification/'


# filelist = sorted(glob.glob(DirIn +'*.npy'))   # All
# hyperpara = '*opti1*lr0.001*decay0.1*batch16*epoch500*'
hyperpara = '*0.0001*0.1*32*200*'
hyperpara = 'Lens*0.001*16'



filelist = sorted(glob.glob(DirIn + hyperpara + '*.hdf5'))
histlist = sorted(glob.glob(DirIn + hyperpara + '*.npy'))

print len(filelist)

for i in range(len(filelist)):
    fileIn = filelist[i]
    histIn = histlist[i]
    loaded_model = load_model(fileIn)
    print fileIn
    history = np.load(histIn)
    print histIn





def load_test():
    img_data_list = []
    labels = []

    # for name in names:
    for labelID in [0, 1]:
        name = names[labelID]
        for img_ind in range(num_files / num_classes):

            input_img = np.load(data_path + name + str(img_ind) + '.npy')
            if np.isnan(input_img).any():
                print labelID, img_ind, ' -- ERROR: NaN'
            else:

                img_data_list.append(input_img)
                labels.append(labelID)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    labels = np.array(labels)
    labels = labels.astype('float32')

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

    X_test = img_data
    y_test = np_utils.to_categorical(labels, num_classes)
    # labels = np.load(Dir1 + Dir2 + Dir3 + 'Test5para.npy')
    # print labels1.shape
    print labels.shape

    # para5 = labels[:,2:]
    np.random.seed(12345)
    shuffleOrder = np.arange(X_test.shape[0])
    np.random.shuffle(shuffleOrder)
    X_test = X_test[shuffleOrder]
    # y_test = para5[shuffleOrder]
    y_test = y_test[shuffleOrder]

    # print y_train[0:10]
    # print y_train[0:10]

    return X_test, y_test

X_test, y_test = load_test()

#------------ Plot accuracy and loss -------------------


plotLossAcc = True
if plotLossAcc:
    import matplotlib.pylab as plt

    epochs =  history[0,:]
    train_loss = history[1,:]
    val_loss = history[2,:]
    train_acc = history[3, :]
    val_acc = history[4, :]


    fig, ax = plt.subplots(2,1, sharex= True, figsize = (7,10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax[0].plot(epochs,train_loss)
    ax[0].plot(epochs,val_loss)
    ax[0].set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[0].legend(['train_loss','val_loss'])

    ax[1].plot(epochs,train_acc)
    ax[1].plot(epochs,val_acc)
    ax[1].set_xlabel('num of Epochs')
    ax[1].set_ylabel('accuracy')
    # ax[1].set_ylim([0,1])
    # ax[1].set_title('Accuracy')
    ax[1].legend(['train_acc','val_acc'], loc=4)

    plt.show()

#------------ Testing -------------------

test_image = X_test[2011, :, :, 0]  #200
Testimg = True

if Testimg:
	# test_image = 255*(test_image - test_image.min())/( test_image.max() - test_image.min())

	# test_image = cv2.resize(test_image, (img_rows, img_cols))

	# test_image=cv2.resize(test_image,(128,128))
	# test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	print (test_image.shape)

	if num_channel==1:
		if K.image_dim_ordering()=='th':
			test_image= np.expand_dims(test_image, axis=0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=3)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)

	else:
		if K.image_dim_ordering()=='th':
			test_image=np.rollaxis(test_image,2,0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)

	# Predicting the test image
	print np.mean(test_image)
	print((loaded_model.predict(test_image)))
	print(loaded_model.predict_classes(test_image))


plotFeaturemaps = False
if plotFeaturemaps:
	def get_featuremaps(model, layer_idx, X_batch):
		get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
		activations = get_activations([X_batch,0])
		return activations

	layer_num = 1
	filter_num=0

	activations = get_featuremaps(loaded_model, int(layer_num),test_image)

	print (np.shape(activations))
	feature_maps = activations[0][0]
	print (np.shape(feature_maps))

	if K.image_dim_ordering()=='th':
		feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
	print (feature_maps.shape)

	fig=plt.figure(figsize=(16,16))
	plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
	plt.savefig("ModelOutputs/featuremapsLayer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.png')

	fig=plt.figure(figsize=(3,3))
	plt.imshow(test_image[0,:,:,0],cmap='gray')

	# num_of_featuremaps=feature_maps.shape[2]
	num_of_featuremaps = 16  # Choosing a smaller subset
	fig=plt.figure(figsize=(16,16))
	plt.title("featuremaps-layer-{}".format(layer_num))
	subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
	for i in range(int(num_of_featuremaps)):
		ax = fig.add_subplot(subplot_num, subplot_num, i+1)
		# ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
		ax.imshow(feature_maps[:,:,i],cmap='gray')
		plt.xticks([])
		plt.yticks([])
		plt.tight_layout()
	# fig.savefig("ModelOutputs/featuremapsLayer.pdf")
	plt.show()



# ---------- Confusion matrix -------------


plotConfusion = True
if plotConfusion:
	from sklearn.metrics import classification_report,confusion_matrix
	import itertools

	Y_pred = loaded_model.predict(X_test)
	print(Y_pred)
	y_pred = np.argmax(Y_pred, axis=1)
	print(y_pred)
	#y_pred = model.predict_classes(X_test)
	#print(y_pred)
	target_names = ['class 0(lensed)', 'class 1(unlensed)']

	print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

	print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


	# Plotting the confusion matrix
	def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(cm)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	# Compute confusion matrix
	cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
	np.set_printoptions(precision=2)
	plt.figure()

	# plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix') # Plot non-normalized confusion matrix
	#plt.figure()
	# Plot normalized confusion matrix
	plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True, title='Normalized confusion matrix')
	#plt.figure()
	plt.show()


# ----- plot True and False positives ------------


TruePositive_cond = np.where( (y_pred ==0) & (y_test[:,0] == 0) )
TruePositives = X_test[TruePositive_cond]

# TrueNegative
FalsePositive_cond = np.where( (y_pred ==0) & (y_test[:,0] == 1) )
FalsePositives = X_test[FalsePositive_cond]

fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
plt.suptitle('True positives')

np.random.seed(1234)
count = 0
for ind in np.random.randint(np.shape(TruePositives)[0], size = 8):

	pixel = TruePositives[ind,:,:,0]
	# for i in range(numPlots):
	ax[count / 4, count % 4].imshow(pixel, cmap=plt.get_cmap('gray'))
	ax[count / 4, count % 4].set_title(str(ind))

	count += 1


fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
plt.suptitle('False positives')

count = 0
for ind in np.random.randint(np.shape(FalsePositives)[0], size = 8):

	pixel = FalsePositives[ind,:,:,0]
	# for i in range(numPlots):
	ax[count / 4, count % 4].imshow(pixel, cmap=plt.get_cmap('gray'))
	ax[count / 4, count % 4].set_title(str(ind))

	count += 1




time_j = time.time()
print(time_j - time_i, 'seconds')
print( (time_j - time_i)/num_files, 'seconds per image' )

# from keras.models import model_from_json
# from keras.models import load_model
#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
#
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# model.save('ModelOutputs/model.hdf5')
# loaded_model=load_model('ModelOutputs/model.hdf5')











#
# # -------------------- Test Checks ---------------------
#
#
# test_imageFile = Dir1+ Dir2 + Dir3 + 'TestData/unlensed_10.npy'
#
# print
# print
# print '------------ predictions --------------'
# print
#
# test_image = np.load(test_imageFile)
# test_image = 255*(test_image - test_image.min())/( test_image.max() - test_image.min())
#
# # test_image = cv2.resize(test_image, (32, 32))
#
# # test_image=cv2.resize(test_image,(128,128))
# # test_image = np.array(test_image)
# test_image = test_image.astype('float32')
# test_image /= 255
# print (test_image.shape)
#
# if num_channel == 1:
# 	if K.image_dim_ordering() == 'th':
# 		test_image = np.expand_dims(test_image, axis=0)
# 		test_image = np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
# 	else:
# 		test_image = np.expand_dims(test_image, axis=3)
# 		test_image = np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
#
# else:
# 	if K.image_dim_ordering() == 'th':
# 		test_image = np.rollaxis(test_image, 2, 0)
# 		test_image = np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
# 	else:
# 		test_image = np.expand_dims(test_image, axis=0)
# 		print (test_image.shape)
#
#
# # Predicting the test image
# print test_imageFile
# print
# print 'class 0(lensed)  |  class 1(unlensed)'
# print((model.predict(test_image)))
# print(model.predict_classes(test_image))
