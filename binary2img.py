"""  Convert .data files files into (20,10,2) .npy files.
Randomly shuffles the files and selects 20 per cent of the files for TestData.
Note: Indices of .npy and .data are NOT the same.
"""


import numpy as np
import matplotlib.pylab as plt
import json
np.set_printoptions(precision=3)
import glob


lid = 1
labels = ['lepton_', 'jet_'][lid]
Dir4 = ['lepton/', 'jet/'][lid]


Dir1 = '/home/nes/Desktop/ConvNetData/atlas/'
Dir2 = 'subimages/'

fileJson = Dir1+Dir2+'*'+labels+'*json'
fileData = Dir1+Dir2+'*'+labels+'*data'

fileInData = sorted(glob.glob(fileData))
fileInJson = sorted(glob.glob(fileJson))

if (len(fileInData) == 0): print 'ERROR: Empty folder'
print 'number of files: ', len(fileInData)

np.random.seed(12345)
alln = np.arange(len(fileInData))
np.random.shuffle(alln)
forTest =  alln[:int(0.2*len(fileInData))]
# forTest = np.random.randint(10000, size = [2000])
testind = 0
trainind = 0

for ind in range(len(fileInData)):
    fileInJsonX = fileInJson[ind]
    fileInDataX = fileInData[ind]


    with open(fileInJsonX, 'r') as f: Jdata = json.load(f)
    print 'eta     phi       r        id        pt'

    print np.array([Jdata['eta'], Jdata['phi'], Jdata['r'], Jdata['id'], Jdata['pt']])

    # pixData = open(fileInDataX).read()
    # pixData = np.fromstring(pixData)

    pixData = np.fromfile(fileInDataX, dtype = np.float32)  # shape (200,)  - reverse of .tobytes() ?
    # print pixeldata.max(), pixeldata.min()
    pixel = np.reshape(pixData, (20, 10, 2), order = 'C')

    if ind in forTest:
        np.save(Dir1 + 'TestData/'+labels+ str(testind), pixel)
        testind+=1
    else:
        np.save(Dir1 + 'TrainingData/'+Dir4+labels+ str(trainind), pixel)
        trainind+=1


# pixel = np.fromfile(fileInData[20], dtype = np.float32).reshape(20,10,2)

# plt.imshow(pixel)

# plt.show()


