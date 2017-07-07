"""
Execute: python2.7 CreateData.py -i <subimages/> -o <DirOut> -n 10
Ex: python2.7 CreateData.py -i rawData/ -o ./ -n 100

Converts .data files files into (20,10,2) .npy files.

Randomly shuffles the files and selects 20 per cent of the files for TestData.
Note: Indices of .npy and .data are NOT the same.
- Possible issue if all the subimage are not created already - mismatch in json and data files

"""
# from __future__ import print_function
#import sys
import numpy as np
import matplotlib.pylab as plt
import json
np.set_printoptions(precision=3)
import glob
import argparse
import os

def CreateFolder(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="DirIn",
                    help="input subimages folder: should have 3 folders ee, mm and tt",
                    metavar="FILE")

parser.add_argument("-o", dest="DirOut",
                    help="output folder: creates 2 folders: TestData and TrainingData",
                    metavar="FILE")

parser.add_argument('-n', metavar='N', type=int, nargs='+', dest = 'NumFiles',
                    help='number of files')

# parser.add_argument("-n", )
#parser.add_argument
args = parser.parse_args()
print
print
print args.DirOut
print args.NumFiles[0]

Dir1 = args.DirIn
Dir2 = args.DirOut


NumFiles = args.NumFiles[0]  #Ideally len(fileInData), use least #files overall



for lid in [0, 1, 2, 3]:
    labels = 'lepton'
    label2 = ['ee', 'mm', 'tt', 'jet'][lid]



    if (lid == 3):
        fileJson = Dir1 +'*'+label2+'*json'
        fileData = Dir1 +'*/*'+label2+'*data'
        # if (len(fileInData) == 0): print fileData, 'ERROR: Empty folder',
        # print 'number of files: ', len(fileInData)

    else:
        fileJson = Dir1 + label2 + '/*' + labels + '*json'
        fileData = Dir1 + label2 + '/*' + labels + '*data'

    fileInData = sorted(glob.glob(fileData))
    fileInJson = sorted(glob.glob(fileJson))

    if (len(fileInData) == 0): print fileData, 'ERROR: Empty folder',
    print 'number of files: ', len(fileInData)

    # print fileInJson[100]
    np.random.seed(12345)


    alln = np.arange(NumFiles)
    np.random.shuffle(alln)
    forTest =  alln[:int(0.2*NumFiles)]
    # forTest = np.random.randint(10000, size = [2000])
    testind = 0
    trainind = 0

    for ind in range(NumFiles):
        # fileInJsonX = fileInJson[ind]
        fileInDataX = fileInData[ind]


        # with open(fileInJsonX, 'r') as f: Jdata = json.load(f)
        # print 'eta     phi       r        id        pt'
        # print np.array([Jdata['eta'], Jdata['phi'], Jdata['r'], Jdata['id'], Jdata['pt']])

        ## pixData = open(fileInDataX).read()
        ## pixData = np.fromstring(pixData)

        pixData = np.fromfile(fileInDataX, dtype = np.float32)  # shape (200,)  - reverse of .tobytes() ?
        # print pixeldata.max(), pixeldata.min()
        pixel = np.reshape(pixData, (20, 10, 2), order = 'C')

        CreateFolder(Dir2 + 'TestData/')
        CreateFolder(Dir2 + 'TrainingData/' + label2+ '/')

        if ind in forTest:
            np.save(Dir2 + 'TestData/'+ label2 + str(testind), pixel)
            testind+=1
            # print pixel.min(), pixel.max(), 'TestData/'+labels+ str(testind)
        else:
            np.save(Dir2 + 'TrainingData/'+ label2+ '/'+ label2+ str(trainind), pixel)
            trainind+=1
            print pixel.min(), pixel.max(), 'TrainingData/'+label2+ '/'+ label2+ str(trainind)


