""" Plots .data files """

import numpy as np
import matplotlib.pylab as plt
plt.set_cmap('Set1_r')
import sys

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--file", dest="fileIn",
                    help="input .data file", metavar="FILE")

#parser.add_argument

args = parser.parse_args()

print
print args.fileIn


pixData = np.fromfile(str(args.fileIn))
print 'shape', np.shape(pixData)

if (np.shape(pixData)[0] == 200):
	pixel = np.reshape(pixData, (20, 10), order = 'C')
	plt.imshow(pixel, )
else: 
	pixel = np.reshape(pixData, (20, 10, 2), order = 'C')
	plt.imshow(pixel)  # Not sure about imshow of 2/3 channels

plt.show()
