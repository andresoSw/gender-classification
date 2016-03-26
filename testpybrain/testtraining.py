from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from os import listdir
from os.path import isfile,join
import numpy as np
import random

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
	print target
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
	classes = {'female':0,'male':1}
	# target = classes[dirname]

	for index,file_name in enumerate(listdir(dirname)):
		data_file = join(dirname,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): continue
		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,SIGNAL_LENGTH,SIGNAL_COUNT)

		result = neuralNetwork.activate(vs[0])
		print file_name, 'classified as', getGender(result[0]), 'with', result[0], ' output'

if __name__ == '__main__':
	#obtaining female_data
	female_data = getTrainingData('female_big')
	female_input = female_data[0]
	female_target =  female_data[1]

	#obtanining male data
	male_data = getTrainingData('male_big')
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


	#building up the network 
	NUMBER_OF_HIDDEN_NEURONS = 200
	LEARNING_RATE = 0.01
	MAX_ITERATIONS = 750

	print "Number of training patterns: %s" %(len(training_dataset))
	print "Input dimension: %s" %(training_dataset.indim)
	print "Output dimension: %s" %(training_dataset.outdim)
	print "Number of hidden Neurons %s" %(NUMBER_OF_HIDDEN_NEURONS)
	print "Learning Rate %s" %(LEARNING_RATE)

	network = buildNetwork( training_dataset.indim,NUMBER_OF_HIDDEN_NEURONS, training_dataset.outdim, bias=True )
	trainer = BackpropTrainer(network,training_dataset,learningrate = LEARNING_RATE, verbose = False)

	#training with training dataset
	for epoch in xrange(0,MAX_ITERATIONS):
		epoch_error = trainer.train()
		print '%s Train Error: %s Train Accuracy: %s' %(epoch,epoch_error,1-epoch_error)

	# Testing network on directories
	male_dir = 'male_big'
	female_dir = 'female_big'
	test_dir = 'test'
	
	print 'Test for %s directory' % (male_dir)

	test_neuralnetwork(male_dir,network)
	print 'Test for %s directory' % (female_dir)
	test_neuralnetwork(female_dir,network)

	print 'Test for %s directory' % (test_dir)
	test_neuralnetwork(test_dir,network)