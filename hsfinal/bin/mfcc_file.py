from os import listdir
from os.path import isfile,join
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import pickle
import numpy as np
import sys

from sys import argv

if (len(argv) < 2) or ('.wav' not in argv[1]):
	print 'to execute run'
	print 'python mfcc_file.py file_name.wav'
	sys.exit()

file_name =  argv[1]
(rate,sig) = wav.read(file_name)
mfcc_feat = mfcc(sig,rate)
for m in mfcc_feat:
	for c in m:
		print c,
	print 
