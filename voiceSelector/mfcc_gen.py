from os import listdir
from os.path import isfile,join
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import pickle
import numpy as np
import sys
from sys import argv

if (len(argv) < 2):
	datasets_dir = 'female_test'
datasets_dir =  argv[1]
new_index = 0
for index,file_name in enumerate(listdir(datasets_dir)):
	if '.wav' not in file_name : continue
	print "%s)=============================================================" %(new_index+1)
	print "Getting MFCCs from dataset: %s" %(file_name)
	data_file = join(datasets_dir,file_name)
	if not isfile(data_file): continue
	(rate,sig) = wav.read(data_file)
	mfcc_feat = mfcc(sig,rate)
	fbank_feat = logfbank(sig,rate)
	output_file = data_file.split('.')[0] + '.mfcc'
	np.savetxt(output_file, mfcc_feat)
	new_index += 1
