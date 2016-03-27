from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import os
from os import listdir
from os.path import isfile,join
import numpy as np
import random

from utilities import extractCommandParams,createRunFolder,writeAsJson,pickleDumpObject,pickleLoadObject
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
	Classifies unlabeled samples given a sample directory 
	@param samplesdir the directory containing the unlabeled samples to be classified
	@param network a trained network
	@return a list containing the classified sample files and
			a list with the respective classifications for each file and
			a list with the respective activation values for each classification
"""
def classifyUnlabeledSamples(samplesdir,network):
	sample_files = []
	classifications = []
	activation_values = []

	for index,file_name in enumerate(listdir(samplesdir)):
		data_file = join(samplesdir,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): 
			continue

		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,SIGNAL_LENGTH,SIGNAL_COUNT)
		result = network.activate(vs[0])
		
		sample_files.append(file_name)
		classifications.append(getGender(result))
		activation_values.append(result)

	return sample_files,classifications,activation_values


"""
	Main training function
"""
def trainGenderClassification(learningRate,hiddenNeurons,bias,maxIterations,femaleDataDir,
							maleDataDir,momentum,signalLength,signalCount,resultsFolder,checkclassdir):

	"""
		Prepating Training and Test datasets
	"""
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


	print '----------------------------------------------------------------'
	print '***** Running Backpropagation Trainer with parameters:\n'
	print '* learningRate   : %s' %(learningRate)
	print '* inputs         : %s' %(training_dataset.indim)
	print '* outputs        : %s' %(training_dataset.outdim)
	print '* hiddenNeurons  : %s' %(hiddenNeurons)
	print '* bias           : %s' %(bias)
	print '* momentum       : %s' %(momentum)
	print '* maxIterations  : %s' %(maxIterations)
	print '* femaleDataDir  : %s' %(femaleDataDir)
	print '* maleDataDir    : %s' %(maleDataDir) 
	print '* signalLength   : %s' %(signalLength)
	print '* signalCount    : %s' %(signalCount)
	print '* resultsFolder  : %s' %(resultsFolder) 
	print '* checkclassdir  : %s' %(checkclassdir)
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
		'inputs':training_dataset.indim,
		'outputs':training_dataset.outdim,
		'hiddenNeurons': hiddenNeurons,
		'bias': bias,
		'momentum': momentum,
		'maxIterations': maxIterations,
		'femaleDataDir': femaleDataDir,
		'maleDataDir': maleDataDir,
		'datasetSize':len(training_dataset),
		'signalLength': signalLength,
		'signalCount':signalCount,
		'resultsFolder':resultsFolder,
		'checkclassdir':checkclassdir
	}
	writeAsJson(input_params,input_params_file,indent=4)

	network = buildNetwork( training_dataset.indim,hiddenNeurons, training_dataset.outdim, bias=bias )
	trainer = BackpropTrainer(network,training_dataset,learningrate = learningRate, momentum=momentum, verbose = False)

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

	if checkclassdir is not None:
		classification_out_filename = 'classification_out.txt'
		classification_out_file = os.path.join(run_path,classification_out_filename)
		"""
		Classification for unlabeled samples
		"""
		print '----------------------------------------------------------------'
		print '**** Classifying samples in %s directory. ' %(checkclassdir)
		print '**   Dumping results in  \"%s\" file ' %(classification_out_filename)
		print '----------------------------------------------------------------'
		
		sample_files,classifications,activation_values = classifyUnlabeledSamples(checkclassdir,network)
		assert((len(sample_files) == len(classifications)) and(len(sample_files)==len(activation_values)))

		with open(classification_out_file, "a") as outfile:
			for samplenum,_ in enumerate(sample_files):
				classification_message = "#%s file \"%s\" classified as <%s>. Activation value: %s " %(samplenum+1,sample_files[samplenum],classifications[samplenum],activation_values[samplenum])
				print classification_message
				outfile.write(classification_message+'\n')
	else:
		print 'No additional samples were specified. <checkclassdir>'
	"""
		Dumping network in pickle file
	"""
	network_result_file = os.path.join(run_path,'network.pickle')

	pickleDumpObject(network,network_result_file)
	network = pickleLoadObject(network_result_file)

"""
	Main program
"""
if __name__ == '__main__':
	arguments = extractCommandParams(sys.argv[1:]) 

	#mandatory args
	learningRate = arguments["learningrate"]
	maxIterations = arguments["iterations"]
	femaleDataDir = arguments["femaledir"]
	maleDataDir = arguments["maledir"]

	#optional args
	DEFAULT_MOMENTUM = 0.
	DEFAULT_BIAS = True
	DEFAULT_SIGNAL_LENGTH = 15
	DEFAULT_SIGNAL_COUNT = 1
	DEFAULT_RESULTS_FOLDER = 'gender-class-runs' #default name of folder where to place the result files
	DEFAULT_CHECK_CLASS_DIR = None
	DEFAULT_MFCC_COEFFICIENTS = 13

	if "momentum" in arguments:
		momentum = arguments["momentum"]
	else:
		momentum = DEFAULT_MOMENTUM
	if "bias" in arguments:
		bias = arguments["bias"]
	else:
		bias = DEFAULT_BIAS
	if "signallength" in arguments:
	  signalLength = arguments["signallength"]
	else:
	  signalLength = DEFAULT_SIGNAL_LENGTH

	#default number of hidden neurons depends on the signal length
	DEFAULT_HIDDEN_NEURONS = DEFAULT_MFCC_COEFFICIENTS*signalLength/2
	if "hiddenneurons" in arguments:
		hiddenNeurons  = arguments["hiddenneurons"]
	else:
		hiddenNeurons = DEFAULT_HIDDEN_NEURONS

	if "signalcount" in arguments:
	  signalCount = arguments["signalcount"]
	else:
	  signalCount = DEFAULT_SIGNAL_COUNT

	if "rfolder" in arguments:
	  resultsFolder = arguments["rfolder"]
	else:
	  resultsFolder = DEFAULT_RESULTS_FOLDER

	if "checkclassdir" in arguments:
	 	checkclassdir = arguments["checkclassdir"]
	else:
	 	checkclassdir = DEFAULT_CHECK_CLASS_DIR

	SIGNAL_LENGTH = signalLength
	SIGNAL_COUNT = signalCount
	trainGenderClassification(learningRate=learningRate,hiddenNeurons=hiddenNeurons,bias=bias,
							maxIterations=maxIterations,femaleDataDir=femaleDataDir,
							maleDataDir=maleDataDir,momentum=momentum,signalLength=signalLength,
							signalCount=signalCount,resultsFolder=resultsFolder,checkclassdir=checkclassdir)
