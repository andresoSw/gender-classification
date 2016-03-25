from os import listdir
from os.path import isfile,join
from neuralnetwork import NeuralNetwork,EstimationError
import numpy as np
import pickle
import random

SIGNAL_LENGTH = 15
SIGNAL_COUNT = 1

def getTrainingData(dirname):
	target = 0
	dir_input = []
	dir_target = []
	if dirname == 'female':
		target = 0
	elif dirname == 'male':
		target = 1 
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
	if output > 0.5:
		return 'male'
	else:
		return 'female'

def getTestData(dirname,neuralNetwork):
	for index,file_name in enumerate(listdir(dirname)):
		data_file = join(dirname,file_name)
		if (not isfile(data_file)) or ('.mfcc' not in file_name): continue
		file_data = np.loadtxt(data_file)
		vs = getVoiceSignal(file_data,SIGNAL_LENGTH,SIGNAL_COUNT)
		values = neuralNetwork.feed_forward(vs)
		result = sum(values)/len(values)
		print file_name, 'classified as', getGender(result), 'with', result[0], ' output'


if __name__ == '__main__':
	#obtaining female_data
	female_data = getTrainingData('female')
	female_input = female_data[0]
	female_target =  female_data[1]
	
	#obtanining male data
	male_data = getTrainingData('male')
	male_input = male_data[0]
	male_target =  male_data[1]

	# combining both data and applying random sort
	training_input = female_input + male_input
	training_target = female_target + male_target
	training_data = zip(training_input, training_target)
	random.shuffle(training_data)
	training_input[:], training_target[:] = zip(*training_data)
	training_target = np.asarray(training_target)


	# Network configuration
 	input_units = len(training_input[0])
 	output_units = 1
	n_hidden = 200
	momentum = 0.9
	neuralNetwork = NeuralNetwork(learning_rate=0.01,n_in=input_units,n_hidden=n_hidden,n_out=output_units, momentum = momentum, activation='sigmoid')
	neuralNetwork.initialize_weights()
	results_file = ''.join(['results_lr',str(neuralNetwork.learning_rate),'_m',str(momentum),'_',str(n_hidden),"hidden",'_gender_classification']+['.out'])

	neuralNetwork.backpropagation(training_input,training_target,maxIterations=500, batch= False,file_name=results_file)
	network_file = ''.join(['trained_lr',str(neuralNetwork.learning_rate),'_m',str(momentum),'_',str(n_hidden),"hidden",'_gender_classification']+['.nn'])
	pickle.dump(neuralNetwork, file(network_file,'wb'))
	nn2 = pickle.load(file(network_file,'rb'))

	# Testing network on directories
	test_dir = 'test'
	male_dir = 'male'
	female_dir = 'female'
	print 'Test for %s directory' % (test_dir)
	getTestData(test_dir,nn2)
	print 'Test for %s directory' % (male_dir)
	getTestData(male_dir,nn2)
	print 'Test for %s directory' % (female_dir)
	getTestData(female_dir,nn2)
