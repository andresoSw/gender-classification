
from os.path import isfile,join
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import pickle
import numpy as np
import sys
from sys import argv
import os


def createMFCC(wavfile,directory='',filename=''):
	if filename=='':
		filename =wavfile.split('.')[0]+'.mfcc'
	(rate,sig) = wav.read(wavfile)
	mfcc_feat = mfcc(sig,rate)
	if directory != '':
		if not os.path.exists(directory):
			os.makedirs(directory)
	output_file = join(directory, filename)
	np.savetxt(output_file, mfcc_feat)
	return output_file

if __name__ == '__main__':
	wavfile = 'test.wav'
	createMFCC(wavfile)
