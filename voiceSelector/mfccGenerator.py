
from os.path import isfile,join
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import pickle
import numpy as np
import sys
from sys import argv


def createMFCC(wavfile,directory=''):
	(rate,sig) = wav.read(wavfile)
	mfcc_feat = mfcc(sig,rate)
	if directory=='':
		filename =wavfile.split('.')[0]+'.mfcc'
		output_file = join(directory, filename)
		print output_file
		np.savetxt(output_file, mfcc_feat)

if __name__ == '__main__':
	wavfile = 'test.wav'
	createMFCC(wavfile)
