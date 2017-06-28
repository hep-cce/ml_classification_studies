"""  Convert .fits files files into (45,45) .npy files.
Randomly shuffles the files and selects 20 per cent of the files for TestData.
Note: Indices of .npy and .fits are NOT the same.
"""


import numpy as np
import matplotlib.pylab as plt
import glob
from astropy.io import fits

lid = 0   # lensed or unlensed - trial classification b/w 2 labels.

Dir1 = '/home/nes/Desktop/ConvNetData/lens/'

Dir2 = ['data_of_lsst/', 'lsst_noiseless_single/', 'lsst_noiseless_stack/'][0]
Dir3 = ['lsst_mocks_single/', 'lsst_mocks_stack/'][0]
Dir4 = ['lensed_outputs/', 'unlensed_outputs/'][lid]
Dir5 = ['0/', '1/'][1]

labels = [ 'lensed_', 'unlensed_'][lid]

# # lsst mocks single
# filesMocksSingleLensed0 = Dir1+'data_of_lsst/lsst_mocks_single/lensed_outputs/0/*gz'
# filesMocksSingleLensed1 = Dir1+'data_of_lsst/lsst_mocks_single/lensed_outputs/1/*gz'
# filesMocksSingleUnlensed0 = Dir1+'data_of_lsst/lsst_mocks_single/unlensed_outputs/0/*gz'
# filesMocksSingleUnlensed1 = Dir1+'data_of_lsst/lsst_mocks_single/unlensed_outputs/1/*gz'
# # lsst mocks stack
# filesMocksStackLensed0 = Dir1+'data_of_lsst/lsst_mocks_stack/lensed_outputs/0/*gz'
# filesMocksStackLensed1 = Dir1+'data_of_lsst/lsst_mocks_stack/lensed_outputs/1/*gz'
# filesMocksStackUnlensed0 = Dir1+'data_of_lsst/lsst_mocks_stack/unlensed_outputs/0/*gz'
# filesMocksStackUnlensed1 = Dir1+'data_of_lsst/lsst_mocks_stack/unlensed_outputs/1/*gz'
# # lsst noiseless
# filesLsstNoiselessSingle = Dir1+'lsst_noiseless_single'
# filesLsstNoiselessStack = Dir1+'lsst_noiseless_stack'
#
# fileIn = filesMocksSingleLensed0

fileIn = Dir1+Dir2+Dir3+Dir4+Dir5+'*gz'
# fileIn = Dir1+Dir2+'*gz'
print 'FITS folder: ', fileIn

fileInData = sorted(glob.glob(fileIn))

if (len(fileInData) == 0): print 'ERROR: Empty folder'
print 'number of files: ', len(fileInData)


# import sys
# sys.exit()

np.random.seed(12345)
alln = np.arange(len(fileInData))
np.random.shuffle(alln)
forTest =  alln[:int(0.2*len(fileInData))]
# forTest = np.random.randint(10000, size = [2000])
testind = 0
trainind = 0

for ind in range(len(fileInData)):
    fileIn = fileInData[ind]
    pixel = fits.open(fileIn, memmap=True)
    print ind
    # np.save(Dir1+'TrainingData/'+Dir2+Dir3+Dir4+Dir5+str(ind), pixel)
    if ind in forTest:
        np.save(Dir1 + 'TestData/'+labels+ str(testind), pixel[0].data)
        testind+=1
    else:
        np.save(Dir1 + 'TrainingData/'+Dir4+labels+ str(trainind), pixel[0].data)
        trainind+=1


# fileIn = fileInData[2000]
# pixel = fits.open(fileIn, memmap=True)
#
# plt.imshow(pixel[0].data)
#
# plt.show()