from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import os
from os import listdir
from os.path import isfile,join
import numpy as np
import random
from random import randint as rand_randint
import math

from utilities import extractCommandParams,createRunFolder,writeAsJson,pickleDumpObject,pickleLoadObject
import sys
from collections import Counter

global SIGNAL_LENGTH,SIGNAL_COUNT
SIGNAL_COUNT = 15
SIGNAL_COUNT = 1

"""
	@param dirname the directory containing mfcc samples
	@param testProportion the percentaje of samples used as validation (test)
	@return the samples of the directory, one set for training and one for validation
		the format of the returned data for each training and test samples is a tuple
		of 3 lists, containg the inputs, targets and mfcc files respectively
"""
def getData(dirname,signalLength,signalCount,testProportion=0.3):
	training_data = ([],[],[])
	test_data = ([],[],[])

	target = 0
	dir_input = []
	dir_target = []
	if dirname.startswith('male'):
		target = 1
	elif dirname.startswith('female'):
		target = 0
	else:
		print 'Warning, directory is not well labeled. Female label will be used by default'
	dir_files = [join(dirname,file) for file in listdir(dirname)]
	mfcc_files = [file for file in dir_files if isfile(file) and '.mfcc' in file]
	random.shuffle(mfcc_files)

	mfcc_num_test_samples = int(math.floor(testProportion*len(mfcc_files)))
	mfcc_num_training_samples = len(mfcc_files) - mfcc_num_test_samples

	#training signal samples are flattened
	for ntraining in xrange(0,mfcc_num_training_samples):
		#get a random mfcc_file and pop it from mfcc file list
		mfcc_file = mfcc_files.pop(rand_randint(0,len(mfcc_files)-1))
		file_data = np.loadtxt(mfcc_file)
		voiceSignal = getVoiceSignal(file_data,signalLength,signalCount)

		#flattening voiceSignal and adding 1 sample per entry to training data
		targets = [target] * len(voiceSignal)
		files = [mfcc_file]*len(voiceSignal)
		training_inputs,training_targets,training_mfccfiles = training_data
		training_inputs.extend(voiceSignal)
		training_targets.extend(targets)
		training_mfccfiles.extend(files)

	#test signal samples are grouped
	for ntest in xrange(0,mfcc_num_test_samples):
		#get a random mfcc_file and pop it from mfcc file list
		mfcc_file = mfcc_files.pop(rand_randint(0,len(mfcc_files)-1))
		file_data = np.loadtxt(mfcc_file)
		voiceSignal = getVoiceSignal(file_data,signalLength,signalCount)


		test_inputs,test_targets,test_mfccfiles = test_data
		test_inputs.append(voiceSignal)
		test_targets.append(target)
		test_mfccfiles.append(mfcc_file)

	#all mfcc files were distributed in training and test samples
	assert(mfcc_files == [])
	return (training_data,test_data)

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
	
	sample_inputs,sample_targets,sample_mfccfiles = sample
	othersample_inputs,othersample_targets,othersample_mfccfiles = othersample

	# combining both samples
	combined_inputs = sample_inputs + othersample_inputs
	combined_targets = sample_targets + othersample_targets
	combined_mfccfiles = sample_mfccfiles + othersample_mfccfiles

	#zipping inputs and ouputs in order to apply random shuffle
	combined_samples = zip(combined_inputs, combined_targets,combined_mfccfiles)
	random.shuffle(combined_samples)
	combined_inputs[:], combined_targets[:],combined_mfccfiles[:] = zip(*combined_samples)
	return (combined_inputs,combined_targets,combined_mfccfiles)

"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@param signalClass a function for signal classification of a set of samples,
			can be either modeActivationValue or avgActivationValue
	@return the percentaje of incorrect classifications
"""
def testOnCustomDataset(dataset,network,signalClass,test_results_file):
	estimated_outputs,targets,activation_values,mfcc_files = getClassificationOnCustomDataset(dataset,network,signalClass)
	assert(len(estimated_outputs) == len(targets))
	assert(len(estimated_outputs) == len(dataset[0]))
	assert(len(estimated_outputs) == len(mfcc_files))

	corrects = 0
	female_corrects = 0
	female_incorrects = 0
	male_corrects = 0
	male_incorrects = 0
	with open(test_results_file,'a') as outfile:
		for samplenum in xrange(0,len(estimated_outputs)):
			correct = estimated_outputs[samplenum]==targets[samplenum]
			info = '%s Expected Class: \"%s\", Classified as: \"%s\". Activation Value: %s \n' %(mfcc_files[samplenum],targets[samplenum],estimated_outputs[samplenum],activation_values[samplenum])
			if correct:
				corrects +=1
				outfile.write('CORRECT '+info)
			elif not correct:
				outfile.write('INCORRECT '+info)

			if (targets[samplenum]=='male' and correct):
				male_corrects += 1
			elif (targets[samplenum]=='male' and not correct):
				male_incorrects += 1
			elif (targets[samplenum]=='female' and correct):
				female_corrects += 1
			elif (targets[samplenum]=='female' and not correct):
				female_incorrects += 1
		if ((female_corrects + male_incorrects) == 0):
			recall = 0
		else:
			recall = female_corrects / float(female_corrects + male_incorrects)

		if ((female_corrects + female_incorrects) == 0):
			precision = 0
		else:
			precision = female_corrects / float(female_corrects + female_incorrects)

		outfile.write('================================\n')
		outfile.write('TOTAL CORRECTS     : %s\n' %(corrects))
		outfile.write('TOTAL INCORRECTS   : %s\n' %(len(estimated_outputs)-corrects))
		outfile.write('CORRECT PERCENTAJE : %5.3f%%\n'%((corrects/float(len(dataset[0])))*100))
		outfile.write('MALE CORRECTS      : %s\n' %(male_corrects))
		outfile.write('MALE INCORRECTS    : %s\n' %(male_incorrects))
		outfile.write('FEMALE CORRECTS    : %s\n' %(female_corrects))
		outfile.write('FEMALE INCORRECTS  : %s\n' %(female_incorrects)) 
		outfile.write('PRECISION          : %s\n' %(precision))
		outfile.write('RECALL             : %s\n' %(recall))


	totalAccuracy = corrects/float(len(dataset[0]))
	totalError = 1-totalAccuracy
	return totalError

"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@return the percentaje of incorrect classifications
"""
def testOnDataset(dataset,network,verbose=False):
	assert(isinstance(dataset,ClassificationDataSet))
	estimated_outputs,targets = getClassificationOnDataset(dataset,network)
	if verbose:
		print 'estimated outputs: ', estimated_outputs
		print 'targets: ', targets
	assert(len(estimated_outputs) == len(targets))
	assert(len(estimated_outputs) == dataset.getLength())

	#if classification matches adds int(True)=1, 0 otherwise int(False)=0 
	corrects = sum([int(estimated_outputs[sample]==targets[sample]) for sample,_ in enumerate(estimated_outputs) ])
	totalAccuracy = corrects/float(dataset.getLength())
	return totalAccuracy

"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@param signalClass a function for signal classification of a set of samples,
			can be either modeActivationValue or avgActivationValue
	@return three lists,the first one containing the estimated outputs,a
			second one with the real targets, and a third one with the mfcc files
"""
def getClassificationOnCustomDataset(dataset,network,signalClass):
	estimated_outputs = []
	targets = []
	activation_values = []
	mfcc_files = []

	inputs = dataset[0]
	outputs = dataset[1]
	files = dataset[2]
	assert((len(inputs) == len(outputs)) and (len(inputs)==len(files)))

	for samplenum in xrange(0,len(inputs)):
		_input = inputs[samplenum]
		target = outputs[samplenum]
		mfcc_file  = files[samplenum]

		estimated_output = signalClass(_input,network)

		estimated_outputs.append(getGender(estimated_output))
		targets.append(getGender(target))
		activation_values.append(estimated_output)
		mfcc_files.append(mfcc_file)

	return estimated_outputs,targets,activation_values,mfcc_files

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
	@param signalClass a function for signal classification of a set of samples,
			can be either modeActivationValue or avgActivationValue
	@return a list containing the classified sample files and
			a list with the respective classifications for each file and
			a list with the respective activation values for each classification
"""
def classifyUnlabeledSamples(samplesdir,network,signalClass,signalLength,signalCount):
	sample_files = []
	classifications = []
	activation_values = []

	for index,file_name in enumerate(listdir(samplesdir)):
		data_file = join(samplesdir,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): 
			continue

		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,signalLength,signalCount)
		result = signalClass(vs,network)
		
		sample_files.append(file_name)
		classifications.append(getGender(result))
		activation_values.append(result)

	return sample_files,classifications,activation_values

"""
	@param inputs a list of input values
	@param network the trained network
	@return the mode of the activation values of each input
"""
def modeActivationValue(inputs,network):
	classes = {'male': 1 ,'female': 0}
	activationValues = [network.activate(_input) for _input in inputs]
	classifications = map(getGender,activationValues)
	classCounter = Counter(classifications)
	mode = classCounter.most_common(1)[0][0]  # Returns the highest occurring item
	return classes[mode]

"""
	@param inputs a list of input values
	@param network the trained network
	@return the average of the activation values of each input
"""
def avgActivationValue(inputs,network):
	activationValues = [network.activate(_input) for _input in inputs]
	avg = sum(activationValues)/float(len(activationValues))
	return avg

"""
	Main training function
"""
def trainGenderClassification(learningRate,hiddenNeurons,bias,maxIterations,femaleDataDir,
							maleDataDir,momentum,signalLength,signalCount,signalClass,
							resultsFolder,checkclassdir):

	"""
		Prepating Training and Test datasets
	"""
	#extracting female and male samples
	female_training_samples,female_test_samples = getData(femaleDataDir,signalLength,signalCount)
	male_training_samples,male_test_samples = getData(maleDataDir,signalLength,signalCount)

	training_inputs,training_targets,training_mfccfiles = combineSamples(female_training_samples,male_training_samples)
	test_inputs,test_targets,test_mfccfiles = combineSamples(female_test_samples,male_test_samples)
	test_dataset = (test_inputs,test_targets,test_mfccfiles)

	assert(len(training_inputs) == len(training_targets))

	#building up pybrain training dataset
	numberofInputs = len(training_inputs[0])

	training_dataset = ClassificationDataSet(numberofInputs, nb_classes=2,class_labels=['Female','Male'])
	for samplenum in xrange(0,len(training_inputs)):
		training_dataset.addSample(training_inputs[samplenum],[training_targets[samplenum]])

	if hiddenNeurons is None:
		hiddenNeurons= (training_dataset.indim + training_dataset.outdim)/2

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
	print '* signalClass    : %s' %(signalClass.__name__)
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
		'signalClass':signalClass.__name__,
		'resultsFolder':resultsFolder,
		'checkclassdir':checkclassdir
	}
	writeAsJson(input_params,input_params_file,indent=4)

	training_dataset_file = os.path.join(run_path,'training_dataset.txt')
	writeAsJson(training_mfccfiles,training_dataset_file,indent=4)

	test_dataset_file = os.path.join(run_path,'test_dataset.txt')
	test_mfccfiles = test_dataset[2] #test_dataset format: (inputs,outputs,mfccfiles)
	writeAsJson(test_mfccfiles,test_dataset_file)

	network = buildNetwork( training_dataset.indim,hiddenNeurons, training_dataset.outdim, bias=bias )
	trainer = BackpropTrainer(network,training_dataset,learningrate = learningRate, momentum=momentum, verbose = False)

	epoch_error = 0 #keeps track of the last error
	tr_accuracy = 0
	tr_error = 0
	#training with training dataset
	for epoch in xrange(0,maxIterations):
		epoch_error = trainer.train()
		tr_accuracy = testOnDataset(training_dataset,network)
		tr_error = 1 - tr_accuracy
		print '%s MSE: %s Train Error: %s Train Accuracy: %s' %(epoch,epoch_error,tr_error,tr_accuracy)
	
	training_error = tr_error
	training_accuracy = 1-training_error

	test_results_file = os.path.join(run_path,'test_results.txt')
	test_error = testOnCustomDataset(test_dataset,network,signalClass,test_results_file)
	test_accuracy = 1-test_error

	print '----------------------------------------------------------------'
	print '**** Training Results:'
	print '----------------------------------------------------------------'
	print '* Training MSE : %s' %(epoch_error)
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
		"MSE":epoch_error,
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
		print '**** Classifying samples in \"%s\" directory. ' %(checkclassdir)
		print '**   Dumping results in  \"%s\" file ' %(classification_out_filename)
		print '----------------------------------------------------------------'
		
		sample_files,classifications,activation_values = classifyUnlabeledSamples(checkclassdir,network,signalClass,signalLength,signalCount)
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

	network.signalCount = signalCount #parameters that need to be stored
	network.signalLength = signalCount
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
	DEFAULT_SIGNAL_CLASS = avgActivationValue
	DEFAULT_RESULTS_FOLDER = 'gender-class-runs' #default name of folder where to place the result files
	DEFAULT_CHECK_CLASS_DIR = None
	DEFAULT_HIDDEN_NEURONS = None #flag, if none the number is based on the input units

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

	if "hiddenneurons" in arguments:
		hiddenNeurons  = arguments["hiddenneurons"]
	else:
		hiddenNeurons = DEFAULT_HIDDEN_NEURONS

	if "signalcount" in arguments:
	  signalCount = arguments["signalcount"]
	else:
	  signalCount = DEFAULT_SIGNAL_COUNT

	if "signalclass" in arguments:
		if arguments["signalclass"] == "mode":
			signalClass = modeActivationValue
		elif arguments["signalclass"] == "avg":
			signalClass = avgActivationValue
	else:
		signalClass = DEFAULT_SIGNAL_CLASS

	if "rfolder" in arguments:
	  resultsFolder = arguments["rfolder"]
	else:
	  resultsFolder = DEFAULT_RESULTS_FOLDER

	if "checkclassdir" in arguments:
	 	checkclassdir = arguments["checkclassdir"]
	else:
	 	checkclassdir = DEFAULT_CHECK_CLASS_DIR

	trainGenderClassification(learningRate=learningRate,hiddenNeurons=hiddenNeurons,bias=bias,
							maxIterations=maxIterations,femaleDataDir=femaleDataDir,
							maleDataDir=maleDataDir,momentum=momentum,signalLength=signalLength,
							signalCount=signalCount,signalClass=signalClass,
							resultsFolder=resultsFolder,checkclassdir=checkclassdir)
