from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import os
from os import listdir
from os.path import isfile,join
import numpy as np
import random

from utilities import extractCommandParams,createRunFolder,writeAsJson
import sys

SIGNAL_LENGTH = 15
SIGNAL_COUNT = 1

def getTrainingData(dirname):
	target = 0
	dir_input = []
	dir_target = []
	if dirname.startswith('male'):
		target = 1
	elif dirname.startswith('female'):
		target = 0
	for index,file_name in enumerate(listdir(dirname)):
		data_file = join(dirname,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): continue
		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,SIGNAL_LENGTH,SIGNAL_COUNT)
		file_target = [target] * len(vs)
		dir_input.extend(vs)
		dir_target.extend(file_target)
	return (dir_input, dir_target)


# Data used
# Length of each signal
# number of signals to be used
def getVoiceSignal(data,length,number=1):
	data_size = len(data)
	voice_part = split(data,number)
	voice_signals = []
	for voice in voice_part:
		voice_data = voice[len(voice)/2:(len(voice)/2)+length]
		signal = [c for v in  voice_data for c in v]
		voice_signals.append(signal)
	return voice_signals

def split(a, n):
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

def getGender(output):
	if output > 0.45:
		return 'male'
	else:
		return 'female'

def test_trainer(dirname,trainer):
	for index,file_name in enumerate(listdir(dirname)):
		data_file = join(dirname,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): continue
		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,SIGNAL_LENGTH,SIGNAL_COUNT)

		"""
		Builds a new test dataset and tests the trained network on it.
		"""
		testdata = ClassificationDataSet(numberofInputs, nb_classes=2,class_labels=['Female','Male'])
		testdata.addSample(vs[0],[target])
		trainer.testOnData(testdata, verbose= True)

def test_neuralnetwork(dirname,neuralNetwork):
	for index,file_name in enumerate(listdir(dirname)):
		data_file = join(dirname,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): continue
		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,SIGNAL_LENGTH,SIGNAL_COUNT)

		########TODO: result must be either the mode or the average of network
		#activation results over vs[i] 
		result = neuralNetwork.activate(vs[0])
		print file_name, 'classified as', getGender(result[0]), 'with', result[0], ' output'

"""
	Main training function
"""
def trainGenderClassification(learningRate,hiddenNeurons,bias,maxIterations,femaleDataDir,
							maleDataDir,signalLength,signalCount,resultsFolder):

	print '----------------------------------------------------------------'
	print '***** Running Backpropagation Trainer with parameters:\n'
	print '* learningRate   : %s' %(learningRate)
	print '* hiddenNeurons  : %s' %(hiddenNeurons)
	print '* bias           : %s' %(bias)
	print '* maxIterations  : %s' %(maxIterations)
	print '* femaleDataDir  : %s' %(femaleDataDir)
	print '* maleDataDir    : %s' %(maleDataDir) 
	print '* signalLength   : %s' %(signalLength)
	print '* signalCount    : %s' %(signalCount)
	print '* resultsFolder  : %s' %(resultsFolder) 
	print '----------------------------------------------------------------'
	"""
		Computing results folder
	"""
	run_path = createRunFolder(resultsFolder=resultsFolder)

	print '**** Dumping results in directory: \"%s\"' %(run_path)
	print '----------------------------------------------------------------'
	input_params_file = os.path.join(run_path,'inputParams.txt')
	input_params = {
		'learningRate': learningRate,
		'hiddenNeurons': hiddenNeurons,
		'bias': bias,
		'maxIterations': maxIterations,
		'femaleDataDir': femaleDataDir,
		'maleDataDir': maleDataDir,
		'signalLength': signalLength,
		'signalCount':signalCount,
		'resultsFolder':resultsFolder
	}
	writeAsJson(input_params,input_params_file)

	#obtaining female_data
	female_data = getTrainingData(femaleDataDir)
	female_input = female_data[0]
	female_target =  female_data[1]

	#obtanining male data
	male_data = getTrainingData(maleDataDir)
	male_input = male_data[0]
	male_target =  male_data[1]

	# combining both data and applying random sort
	training_input = female_input + male_input
	training_target = female_target + male_target
	training_data = zip(training_input, training_target)
	random.shuffle(training_data)
	training_input[:], training_target[:] = zip(*training_data)

	assert(len(training_input) == len(training_target))

	#building up pybrain training dataset
	numberofInputs = len(training_input[0])
	training_dataset = ClassificationDataSet(numberofInputs, nb_classes=2,class_labels=['Female','Male'])
	for samplenum in xrange(0,len(training_input)):
		training_dataset.addSample(training_input[samplenum],[training_target[samplenum]])

	print "Number of training patterns: %s" %(len(training_dataset))
	print "Input dimension: %s" %(training_dataset.indim)
	print "Output dimension: %s" %(training_dataset.outdim)
	print "Number of hidden Neurons %s" %(hiddenNeurons)
	print "Learning Rate %s" %(learningRate)

	network = buildNetwork( training_dataset.indim,hiddenNeurons, training_dataset.outdim, bias=bias )
	trainer = BackpropTrainer(network,training_dataset,learningrate = learningRate, verbose = False)

	#training with training dataset
	for epoch in xrange(0,maxIterations):
		epoch_error = trainer.train()
		print '%s Train Error: %s Train Accuracy: %s' %(epoch,epoch_error,1-epoch_error)

	# Testing network on directories
	male_dir = maleDataDir
	female_dir = femaleDataDir
	test_dir = 'test'
	
	print 'Test for %s directory' % (male_dir)

	test_neuralnetwork(male_dir,network)
	print 'Test for %s directory' % (female_dir)
	test_neuralnetwork(female_dir,network)

	print 'Test for %s directory' % (test_dir)
	test_neuralnetwork(test_dir,network)
	 

if __name__ == '__main__':
	arguments = extractCommandParams(sys.argv[1:]) 

	#mandatory args
	learningRate = arguments["learningrate"]
	hiddenNeurons = arguments["hiddenneurons"]
	bias = arguments["bias"]
	maxIterations = arguments["iterations"]
	femaleDataDir = arguments["femaledir"]
	maleDataDir = arguments["maledir"]
	global SIGNAL_LENGTH,SIGNAL_COUNT

	#optional args
	DEFAULT_SIGNAL_LENGTH = 15
	DEFAULT_SIGNAL_COUNT = 1
	DEFAULT_RESULTS_FOLDER = 'gender-class-runs' #default name of folder where to place the result files

	if "signallength" in arguments:
	  signalLength = arguments["signallength"]
	else:
	  signalLength = DEFAULT_SIGNAL_LENGTH

	if "signalcount" in arguments:
	  signalCount = arguments["signalcount"]
	else:
	  signalCount = DEFAULT_SIGNAL_COUNT

	if "rfolder" in arguments:
	  resultsFolder = arguments["rfolder"]
	else:
	  resultsFolder = DEFAULT_RESULTS_FOLDER

	SIGNAL_LENGTH = signalLength
	SIGNAL_COUNT = signalCount
	trainGenderClassification(learningRate=learningRate,hiddenNeurons=hiddenNeurons,bias=bias,
							maxIterations=maxIterations,femaleDataDir=femaleDataDir,
							maleDataDir=maleDataDir,signalLength=signalLength,
							signalCount=signalCount,resultsFolder=resultsFolder)
