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

global SIGNAL_LENGTH,SIGNAL_COUNT
SIGNAL_COUNT = 15
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
	Randomly splits the full dataset into training and test data sets
	@param fulldataset the dataset to be splited
	@param testSeparationProportion the percentaje of samples to be included 
			within the valiadtion (test) dataset
	@return a tuple containg the test and training datasets respectively
"""
#Randomly split the full dataset into training and test data sets
def splitDatasetWithProportion(fulldataset,testSeparationProportion):
	#Randomly split the full dataset into training test data sets
	test_dataset_temp, training_dataset_temp = fulldataset.splitWithProportion(testSeparationProportion)

	training_dataset = ClassificationDataSet(fulldataset.indim, nb_classes=2,class_labels=['Female','Male'])
	for samplenum in xrange(0,training_dataset_temp.getLength()):
		training_dataset.addSample(training_dataset_temp.getSample(samplenum)[0],training_dataset_temp.getSample(samplenum)[1])

	test_dataset = ClassificationDataSet(fulldataset.indim, nb_classes=2,class_labels=['Female','Male'])
	for samplenum in xrange(0, test_dataset_temp.getLength()):
		test_dataset.addSample( test_dataset_temp.getSample(samplenum)[0],test_dataset_temp.getSample(samplenum)[1] )

	return test_dataset,training_dataset

"""
	Return the combination of two given samples
	@param sample a sample to be combined
	@param othersample a sample to be combined
	@return the combined sample
"""
def combineSamples(sample,othersample):
	
	sample_inputs,sample_targets = sample
	othersample_inputs,othersample_targets = othersample

	# combining both samples
	combined_inputs = sample_inputs + othersample_inputs
	combined_targets = sample_targets + othersample_targets

	#zipping inputs and ouputs in order to apply random shuffle
	combined_samples = zip(combined_inputs, combined_targets)
	random.shuffle(combined_samples)
	combined_inputs[:], combined_targets[:] = zip(*combined_samples)
	return (combined_inputs,combined_targets)

"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@return the percentaje of incorrect classifications
"""
def testOnDataset(dataset,network,verbose=False):
	estimated_outputs,targets = getClassificationOnDataset(dataset,network)
	if verbose:
		print 'estimated outputs: ', estimated_outputs
		print 'targets: ', targets
	assert(len(estimated_outputs) == len(targets))
	assert(len(estimated_outputs) == dataset.getLength())

	#if classification matches adds int(True)=1, 0 otherwise int(False)=0 
	corrects = sum([int(estimated_outputs[sample]==targets[sample]) for sample,_ in enumerate(estimated_outputs) ])
	totalAccuracy = corrects/float(dataset.getLength())
	totalError = 1-totalAccuracy
	return totalError

"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@return two lists,the first one containing the estimated outputs and a
			second one with the real targets
"""
def getClassificationOnDataset(dataset,network):
	estimated_outputs = []
	targets = []

	for samplenum in xrange(0,dataset.getLength()):
		_input = dataset.getSample(samplenum)[0]
		target = dataset.getSample(samplenum)[1]

		estimated_output = network.activate(_input)

		estimated_outputs.append(getGender(estimated_output))
		targets.append(getGender(target))

	return estimated_outputs,targets


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
	writeAsJson(input_params,input_params_file,indent=4)

	#extracting female and male samples
	female_samples = getTrainingData(femaleDataDir)
	male_samples = getTrainingData(maleDataDir)

	training_inputs,training_targets = combineSamples(female_samples,male_samples)

	assert(len(training_inputs) == len(training_targets))

	#building up pybrain training dataset
	numberofInputs = len(training_inputs[0])
	testProportion = 0.3 #30% of the dataset samples will be used for validation 

	full_dataset = ClassificationDataSet(numberofInputs, nb_classes=2,class_labels=['Female','Male'])
	for samplenum in xrange(0,len(training_inputs)):
		full_dataset.addSample(training_inputs[samplenum],[training_targets[samplenum]])

	#Randomly split the full dataset into training and test data sets
	test_dataset, training_dataset = splitDatasetWithProportion(full_dataset,testProportion) 
	
	print "Number of training patterns: %s" %(len(training_dataset))
	print "Input dimension: %s" %(training_dataset.indim)
	print "Output dimension: %s" %(training_dataset.outdim)
	print "Number of hidden Neurons %s" %(hiddenNeurons)
	print "Learning Rate %s" %(learningRate)

	network = buildNetwork( training_dataset.indim,hiddenNeurons, training_dataset.outdim, bias=bias )
	trainer = BackpropTrainer(network,training_dataset,learningrate = learningRate, verbose = False)

	epoch_error = 0 #keeps track of the last error
	#training with training dataset
	for epoch in xrange(0,maxIterations):
		epoch_error = trainer.train()
		print '%s Train Error: %s Train Accuracy: %s' %(epoch,epoch_error,1-epoch_error)
	
	training_error = epoch_error
	training_accuracy = 1-training_error

	test_error = testOnDataset(test_dataset,network)
	test_accuracy = 1-test_error

	print '----------------------------------------------------------------'
	print '**** Training Results:'
	print '----------------------------------------------------------------'
	print '* Training Dataset Accuracy: %s' %(training_accuracy)
	print '* Training Dataset Error: %s' %(training_error)
	print '* Test Dataset Accuracy: %s' %(test_accuracy)
	print '* Test Dataset Error %s' %(test_error)
	print '----------------------------------------------------------------'
	print 'Dumping Results in \"results_out.txt\" file '
	print '----------------------------------------------------------------'
	
	"""
		Dumping results in out file
	"""
	results_out_file = os.path.join(run_path,'results_out.txt')
	results_out = {
		"training_accuracy":training_accuracy, #w training dataset
		"training_error":training_error, #w training dataset
		"test_accuracy":test_accuracy, #w test dataset
		"test_error":test_error #w test dataset
	}
	writeAsJson(results_out,results_out_file,indent=4)

if __name__ == '__main__':
	arguments = extractCommandParams(sys.argv[1:]) 

	#mandatory args
	learningRate = arguments["learningrate"]
	hiddenNeurons = arguments["hiddenneurons"]
	bias = arguments["bias"]
	maxIterations = arguments["iterations"]
	femaleDataDir = arguments["femaledir"]
	maleDataDir = arguments["maledir"]

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
