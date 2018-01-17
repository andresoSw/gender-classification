from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
from random import randint as rand_randint
import math
import time
import psutil
from sys import getsizeof
import sqlite3

from utilities import extractCommandParams, createRunFolder, writeAsJson, pickleDumpObject, pickleLoadObject
import sys
from collections import Counter

from python_speech_features import mfcc
import scipy.io.wavfile as wav

memoryLog=0
CONVERTED_NETWORK_FILE = "tmpFiles/tmpNetworkFile"

def testFileOnNetwork(file,networkSelectedName,signalClass):
    #--------Fetching the network-------#
    rows = getAllNetworksFromDB('mydb')#fetch all networks from db
    row = [row for row in rows if row[1] == networkSelectedName][0]  # Fetching the network from the propper row in the db
    signalLength = row[4]
    signalSampleBuffer = row[5]
    processType=row[6]
    blob = row[7]
    with open(CONVERTED_NETWORK_FILE, 'wb') as output_file:
        output_file.write(blob)
    deserializedNetwork = pickleLoadObject(CONVERTED_NETWORK_FILE)

    # --------building the signal from file-------#
    (rate, sig) = wav.read('tmpFiles/'+file)
    if (len(sig) < signalLength):
        return "file tested too short to learn"
    voiceSignal = getVoiceSignal(sig, rate, signalLength, signalSampleBuffer, processType)  # using wave.rea
    if (len(voiceSignal) == 0):
        return "you are not supposed to arrive here"

    # --------Running the file on the network-------#
    estimated_output = signalClass(voiceSignal, deserializedNetwork)
    prediction = getGender(estimated_output)

    return prediction

def trainNewNetwork(description,learningRate,maxIterations,processType,signalLength,signalSampleBuffer):
    commandParams = ['-l',learningRate,'-i',maxIterations,
                     '-f','female','-m','male','-p',processType,
                     '--rfolder','my-classification-results','--signallength',signalLength,
                     '--signalSampleBuffer',signalSampleBuffer];

    performanceResult = main(commandParams)

    male_training_precision=    performanceResult[0]
    female_training_precision=  performanceResult[1]
    male_test_precision=        performanceResult[2]
    male_test_recall=           performanceResult[3]
    female_test_precision=      performanceResult[4]
    female_test_recall=         performanceResult[5]

    insertNetworkToDB('mydb','tmpFiles/network_saved.p',description,learningRate,maxIterations,processType,signalLength,signalSampleBuffer,
                                                                                                                                          male_training_precision,
                                                                                                                                          female_training_precision,
                                                                                                                                          male_test_precision,
                                                                                                                                          male_test_recall,
                                                                                                                                          female_test_precision,
                                                                                                                                          female_test_recall)
    return "success or failure"

def insertNetworkToDB(dbName,networkPath,description,learningRate,maxIterations,processType,signalLength,signalSampleBuffer,male_training_precision,
                                                                                                                            female_training_precision,
                                                                                                                            male_test_precision,
                                                                                                                            male_test_recall,
                                                                                                                            female_test_precision,
                                                                                                                            female_test_recall):
    with open('tmpFiles/network_saved.p', 'r') as input_file:
        content = input_file.read()
    blob = sqlite3.Binary(content)

    db = connectToDB(dbName)
    cur = db.cursor()
    sql = ''' INSERT INTO TRAINED_NEURAL_NETWORKS(  description,
                                                    learningRate,
                                                    maxIterations,
                                                    signal_length,
                                                    signal_sample_buffer,
                                                    process_type,
                                                    network,
                                                    male_training_precision,
                                                    female_training_precision,
                                                    male_test_precision,
                                                    male_test_recall,
                                                    female_test_precision,
                                                    female_test_recall)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    new_network = (description,float(learningRate),int(maxIterations),int(signalLength),int(signalSampleBuffer),processType,blob,
                   male_training_precision, female_training_precision, male_test_precision, male_test_recall, female_test_precision, female_test_recall);
    cur.execute(sql,new_network);
    db.commit();
    db.close()

def deep_getsizeof(o,name,indentCount):

    curSize = getsizeof(o);
    sum = curSize;

    for indent in xrange(0, indentCount):
        sys.stdout.write('\t')
    print '%s, %s bytes' % (name, curSize)

    #TODO: check what happens to array also
    if(isinstance(o,dict)): #if it is a dictionary
        items = o.items()
        items = dict(items)
    else:                   #get all attributes
        items = getattr(o, '__dict__', None) #TODO:check out what happens if you do training_dataset.items()

    if(items!=None):
        for attr, value in items.iteritems():
            attrSize=deep_getsizeof(value,attr,indentCount+1);
            sum+=attrSize;

    return sum;

def connectToDB(path):
    db = sqlite3.connect(path)
    return db

def closeDB(db):
    db.commit;
    db.close

def getAllNetworksFromDB(dbParam):
    db = connectToDB(dbParam)
    cur = db.cursor()
    cur.execute('''SELECT           id,
                                    description,
                                    learningRate,
                                    maxIterations,
                                    signal_length,
                                    signal_sample_buffer,
                                    process_type,
                                    network
                                        FROM TRAINED_NEURAL_NETWORKS''')
    rows = cur.fetchall();
    closeDB(db)

    return rows



def getData(dirname, signalLength, signalSampleBuffer,processType, testProportion=0.2):
    training_data = ([], [], [])
    test_data = ([], [], [])

    target = 0
    dir_input = []
    dir_target = []
    if dirname.startswith('male'):
        target = 1
    elif dirname.startswith('female'):
        target = 0
    else:
        print 'Warning, directory is not well labeled. Female label will be used by default'
    dir_files = [join(dirname, file) for file in listdir(dirname)]
    wav_files = [file for file in dir_files if isfile(file) and '.wav' in file]
    random.shuffle(wav_files)

    wav_num_test_samples = int(math.floor(testProportion * len(wav_files)))
    wav_num_training_samples = len(wav_files) - wav_num_test_samples


    # training signal samples are flattened
    for ntraining in xrange(0, wav_num_training_samples):
        # get a random mfcc_file and pop it from mfcc file list
        wav_file = wav_files.pop(rand_randint(0, len(wav_files) - 1))
        #At this point I have my wav file as pure as it gets

        print "Training loading..."+str(round(float(ntraining)/wav_num_training_samples,2)*100)+"% "+str(ntraining)+"/"+str(wav_num_training_samples)+" free -m:"+str((psutil.virtual_memory()[1]) / (1024 * 1024))
        (rate, sig) = wav.read(wav_file)

        #Note to self: vadIndication will be shorter then sig since we skip some of the data
        if(len(sig)<signalLength):
            print wav_file+" too short to learn"
            continue

        voiceSignal = getVoiceSignal(sig, rate, signalLength, signalSampleBuffer ,processType)  # using wave.rea

        if(len(voiceSignal)==0):
            print wav_file + " you are not supposed to arrive here"
            continue

        # # Testing Area to plot the waveRead and sig to find the more accurate one to use in VAD starts here
        # plt.figure(1)
        # plt.subplot(2,2,1)
        # plt.plot(sig)
        # plt.xlabel('Speech signal')
        #
        #
        # plt.subplot(2, 2,3)
        # plt.plot(vadIndication,'g')
        #
        # #Testing area for noise trimming
        # sigThresholded = sig[np.logical_or(sig>1000,sig<-1000)]
        # tmp, vadIndicationThresholded = getVoiceSignal(sigThresholded, rate, signalLength, signalCount);
        # vadIndicationThresholded = map(lambda x: x * max(sigThresholded), vadIndicationThresholded)
        #
        # plt.subplot(2, 2, 2)
        # plt.plot(sigThresholded)
        # plt.xlabel('Speech signal thresholded')
        #
        # plt.subplot(2, 2, 4)
        # plt.plot(vadIndicationThresholded, 'g')
        #
        # #
        # plt.show()
        # plt.close()
        # # To here

        # flattening voiceSignal and adding 1 sample per entry to training data
        targets = [target] * len(voiceSignal)
        files = [wav_file] * len(voiceSignal)
        training_inputs, training_targets, training_mfccfiles = training_data
        training_inputs.extend(voiceSignal)
        training_targets.extend(targets)
        training_mfccfiles.extend(files)#TODO: check what the part of mfccfiles

    # test signal samples are grouped
    for ntest in xrange(0, wav_num_test_samples):
        # get a random mfcc_file and pop it from mfcc file list
        print "Test loading..." + str(round(float(ntest)/wav_num_test_samples,2)*100) + "% " + str(ntest) + "/" + str(wav_num_test_samples)

        wav_file = wav_files.pop(rand_randint(0, len(wav_files) - 1))

        (rate, sig) = wav.read(wav_file)

        if (len(sig) < signalLength):
            print wav_file + " too short to learn"
            continue
        
        voiceSignal = getVoiceSignal(sig, rate, signalLength, signalSampleBuffer,processType)
        if (len(voiceSignal) == 0):
            print wav_file + " you are not supposed to arrive here"
            continue

        test_inputs, test_targets, test_mfccfiles = test_data
        test_inputs.append(voiceSignal)
        test_targets.append(target)
        test_mfccfiles.append(wav_file)


    # all mfcc files were distributed in training and test samples
    assert (wav_files == [])
    return (training_data, test_data)

# Data used
# Length of each signal
# number of signals to be used
def getVoiceSignal(data, rate, length, signalSampleBuffer,processType):

    data_size = len(data)
    voice_signals = []

    for index in enumerate(data[0:data_size-length+1]):
        if (index[0]%signalSampleBuffer==0):
            voice_data=data[index[0]:index[0]+length]

            if (processType=='mfcc'):
                mfcc_feat = mfcc(voice_data,16000,winlen=(float(length)/16000)) #winlen calculation is used to extract 13 params exactly instead of a couple
                signal = [c for v in mfcc_feat for c in v] #for MFCC #takes a 2d array(13*15) and create a one-dimensional(195)
                voice_signals.append(signal) #for MFCC
            else:
                if (processType=='wav'):
                    voice_signals.append(voice_data)  # for PURE_WAV
                else:
                    print 'unknown processType.. ENTER mfcc or wav as parameter'

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


# Randomly split the full dataset into training and test data sets
def splitDatasetWithProportion(fulldataset, testSeparationProportion):
    # Randomly split the full dataset into training test data sets
    test_dataset_temp, training_dataset_temp = fulldataset.splitWithProportion(testSeparationProportion)

    training_dataset = ClassificationDataSet(fulldataset.indim, nb_classes=2, class_labels=['Female', 'Male'])
    for samplenum in xrange(0, training_dataset_temp.getLength()):
        training_dataset.addSample(training_dataset_temp.getSample(samplenum)[0],
                                   training_dataset_temp.getSample(samplenum)[1])

    test_dataset = ClassificationDataSet(fulldataset.indim, nb_classes=2, class_labels=['Female', 'Male'])
    for samplenum in xrange(0, test_dataset_temp.getLength()):
        test_dataset.addSample(test_dataset_temp.getSample(samplenum)[0], test_dataset_temp.getSample(samplenum)[1])

    return test_dataset, training_dataset


"""
	Return the combination of two given samples
	@param sample a sample to be combined
	@param othersample a sample to be combined
	@return the combined sample
"""


def combineSamples(sample, othersample): #TODO: The use of this method is not understood
    sample_inputs, sample_targets, sample_mfccfiles = sample
    othersample_inputs, othersample_targets, othersample_mfccfiles = othersample

    # combining both samples
    combined_inputs = sample_inputs + othersample_inputs
    combined_targets = sample_targets + othersample_targets
    combined_mfccfiles = sample_mfccfiles + othersample_mfccfiles

    # zipping inputs and ouputs in order to apply random shuffle
    combined_samples = zip(combined_inputs, combined_targets, combined_mfccfiles)
    random.shuffle(combined_samples)
    combined_inputs[:], combined_targets[:], combined_mfccfiles[:] = zip(*combined_samples)
    return (combined_inputs, combined_targets, combined_mfccfiles)


"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@param signalClass a function for signal classification of a set of samples,
			can be either modeActivationValue or avgActivationValue
	@return the percentaje of incorrect classifications
"""


def testOnCustomDataset(dataset, network, signalClass, test_results_file, performenceResult):
    estimated_outputs, targets, activation_values, mfcc_files = getClassificationOnCustomDataset(dataset, network,
                                                                                                 signalClass)
    assert (len(estimated_outputs) == len(targets))
    assert (len(estimated_outputs) == len(dataset[0]))
    assert (len(estimated_outputs) == len(mfcc_files))

    corrects = 0
    female_corrects = 0
    female_incorrects = 0
    male_corrects = 0
    male_incorrects = 0
    with open(test_results_file, 'a') as outfile:
        for samplenum in xrange(0, len(estimated_outputs)):
            correct = estimated_outputs[samplenum] == targets[samplenum]
            info = '%s Expected Class: \"%s\", Classified as: \"%s\". Activation Value: %s \n' % (
            mfcc_files[samplenum], targets[samplenum], estimated_outputs[samplenum], activation_values[samplenum])
            if correct:
                corrects += 1
                outfile.write('CORRECT ' + info)
            elif not correct:
                outfile.write('INCORRECT ' + info)

            if (targets[samplenum] == 'male' and correct):
                male_corrects += 1
            elif (targets[samplenum] == 'male' and not correct):
                male_incorrects += 1
            elif (targets[samplenum] == 'female' and correct):
                female_corrects += 1
            elif (targets[samplenum] == 'female' and not correct):
                female_incorrects += 1


        # PRECISION
        if ((female_corrects + male_incorrects) == 0):
            female_precision = 0
        else:
            female_precision = female_corrects / float(female_corrects + male_incorrects)

        if ((male_corrects + female_incorrects) == 0):
            male_precision = 0
        else:
            male_precision = male_corrects / float(male_corrects + female_incorrects)

        # RECALL
        if ((female_corrects + female_incorrects) == 0):
            female_recall = 0
        else:
            female_recall = female_corrects / float(female_corrects + female_incorrects)

        if ((male_corrects + male_incorrects) == 0):
            male_recall = 0
        else:
            male_recall = male_corrects / float(male_corrects + male_incorrects)

        performenceResult.extend([male_precision,male_recall,female_precision,female_recall]);
        outfile.write('================================\n')
        outfile.write('TOTAL CORRECTS     : %s\n' % (corrects))
        outfile.write('TOTAL INCORRECTS   : %s\n' % (len(estimated_outputs) - corrects))
        outfile.write('CORRECT PERCENTAJE : %5.3f%%\n' % ((corrects / float(len(dataset[0]))) * 100))
        outfile.write('MALE CORRECTS      : %s\n' % (male_corrects))
        outfile.write('MALE INCORRECTS    : %s\n' % (male_incorrects))
        outfile.write('FEMALE CORRECTS    : %s\n' % (female_corrects))
        outfile.write('FEMALE INCORRECTS  : %s\n' % (female_incorrects))
        outfile.write('PRECISION FEMALE   : %s\n' % (female_precision))
        outfile.write('RECALL FEMALE      : %s\n' % (female_recall))
        outfile.write('PRECISION MALE     : %s\n' % (male_precision))
        outfile.write('RECALL MALE        : %s\n' % (male_recall))

        print '----------------------------------------------------------------'
        print '**** Test Results:'
        print '----------------------------------------------------------------'
        print '* MALE CORRECTS PERCENTAJE: %s' % (male_corrects/float(male_corrects+male_incorrects))
        print '* FEMALE CORRECTS PERCENTAJE: %s' % (female_corrects/float(female_corrects + female_incorrects))

        print 'PRECISION - #relevant_found/#found [How many selected items are relevant?]'
        print ('PRECISION FEMALE   : %s' % (female_precision))
        print ('PRECISION MALE     : %s' % (male_precision))
        print '\nRECALL - #relevant_found/#relevant [Hoy many relevant items are selected?]'
        print ('RECALL FEMALE      : %s' % (female_recall))
        print ('RECALL MALE        : %s' % (male_recall))

    totalAccuracy = corrects / float(len(dataset[0]))
    totalError = 1 - totalAccuracy
    return totalError


"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@return the percentaje of incorrect classifications
"""


def testOnDataset(dataset, network, verbose=False):
    assert (isinstance(dataset, ClassificationDataSet))
    estimated_outputs, targets = getClassificationOnDataset(dataset, network)
    if verbose:
        print 'estimated outputs: ', estimated_outputs
        print 'targets: ', targets
    assert (len(estimated_outputs) == len(targets))
    assert (len(estimated_outputs) == dataset.getLength())

    # if classification matches adds int(True)=1, 0 otherwise int(False)=0
    corrects_male = sum([int((estimated_outputs[sample] == targets[sample] and targets[sample]=='male')) for sample, _ in enumerate(estimated_outputs)])
    corrects_female = sum([int((estimated_outputs[sample] == targets[sample] and targets[sample]=='female')) for sample, _ in enumerate(estimated_outputs)])
    totalMale = sum([int(targets[sample] == 'male') for sample, _ in enumerate(estimated_outputs)])
    totalMale = sum([int(targets[sample] == 'male') for sample, _ in enumerate(estimated_outputs)])
    totalFemale = sum([int(targets[sample] == 'female') for sample, _ in enumerate(estimated_outputs)])

    totalAccuracyMales = corrects_male / float(totalMale)
    totalAccuracyFemales = corrects_female / float(totalFemale)

    return totalAccuracyMales, totalAccuracyFemales


"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@param signalClass a function for signal classification of a set of samples,
			can be either modeActivationValue or avgActivationValue
	@return three lists,the first one containing the estimated outputs,a
			second one with the real targets, and a third one with the mfcc files
"""


def getClassificationOnCustomDataset(dataset, network, signalClass):
    estimated_outputs = []
    targets = []
    activation_values = []
    mfcc_files = []

    inputs = dataset[0]
    outputs = dataset[1]
    files = dataset[2]
    assert ((len(inputs) == len(outputs)) and (len(inputs) == len(files)))

    for samplenum in xrange(0, len(inputs)):
        _input = inputs[samplenum]
        target = outputs[samplenum]
        mfcc_file = files[samplenum]

        estimated_output = signalClass(_input, network)

        estimated_outputs.append(getGender(estimated_output))
        targets.append(getGender(target))
        activation_values.append(estimated_output)
        mfcc_files.append(mfcc_file)

    return estimated_outputs, targets, activation_values, mfcc_files


"""
	@param dataset the validation (test) dataset
	@param network the trained network
	@return two lists,the first one containing the estimated outputs and a
			second one with the real targets
"""


def getClassificationOnDataset(dataset, network):
    estimated_outputs = []
    targets = []

    for samplenum in xrange(0, dataset.getLength()):
        _input = dataset.getSample(samplenum)[0]
        target = dataset.getSample(samplenum)[1]

        estimated_output = network.activate(_input)

        estimated_outputs.append(getGender(estimated_output))
        targets.append(getGender(target))

    return estimated_outputs, targets


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


def classifyUnlabeledSamples(samplesdir, network, signalClass, signalLength, signalSampleBuffer):
    sample_files = []
    classifications = []
    activation_values = []

    for index, file_name in enumerate(listdir(samplesdir)):
        data_file = join(samplesdir, file_name)
        if (not isfile(data_file)) or ('.mfcc' not in file_name):
            continue

        file_data = np.loadtxt(data_file)
        vs = getVoiceSignal(file_data, signalLength, signalSampleBuffer)
        result = signalClass(vs, network)

        sample_files.append(file_name)
        classifications.append(getGender(result))
        activation_values.append(result)

    return sample_files, classifications, activation_values

"""
	@param inputs a list of input values
	@param network the trained network
	@return the mode of the activation values of each input
"""


def modeActivationValue(inputs, network):
    classes = {'male': 1, 'female': 0}
    activationValues = [network.activate(_input) for _input in inputs]
    classifications = map(getGender, activationValues)
    classCounter = Counter(classifications)
    mode = classCounter.most_common(1)[0][0]  # Returns the highest occurring item
    return classes[mode]


"""
	@param inputs a list of input values
	@param network the trained network
	@return the average of the activation values of each input
"""


def avgActivationValue(inputs, network):
    activationValues = [network.activate(_input) for _input in inputs]
    avg = sum(activationValues) / float(len(activationValues))
    return avg


"""
	Main training function
"""

def startMemoryCounter():
    global memoryLog
    memoryLog = (psutil.virtual_memory()[1]) / (1024 * 1024)  # Available memory in MB

def printMemoryDiffFromNow(messageToPrint):
    global memoryLog
    availableMem = (psutil.virtual_memory()[1]) / (1024 * 1024)  # Available memory in MB
    print '%s \tMB used: %s'%(memoryLog-availableMem,messageToPrint)
    memoryLog=0

def trainGenderClassification(learningRate, hiddenNeurons, bias, maxIterations, femaleDataDir,
                              maleDataDir, momentum, signalLength, signalClass,
                              resultsFolder, checkclassdir, signalSampleBuffer,processType):
    performenceResult = []
    """
		Prepating Training and Test datasets
	"""
    # extracting female and male samples
    startMemoryCounter()
    female_training_samples, female_test_samples = getData(femaleDataDir, signalLength, signalSampleBuffer,processType)
    print '* female training samples: %s' % (len(female_training_samples[0]))
    print '* female test samples: %s' % (len(female_test_samples[0]))
    printMemoryDiffFromNow("getData(femaleDataDir,...)")

    startMemoryCounter()
    male_training_samples, male_test_samples = getData(maleDataDir, signalLength, signalSampleBuffer,processType)
    print '* male training samples: %s' % (len(male_training_samples[0]))
    print '* male test samples: %s' % (len(male_test_samples[0]))
    printMemoryDiffFromNow("getData(maleDataDir,...)")
    print "* "

    startMemoryCounter()
    training_inputs, training_targets, training_mfccfiles = combineSamples(female_training_samples,
                                                                           male_training_samples)
    printMemoryDiffFromNow("combineSamples(female_training_samples,male_training_samples)")

    startMemoryCounter()
    test_inputs, test_targets, test_mfccfiles = combineSamples(female_test_samples, male_test_samples)
    printMemoryDiffFromNow("combineSamples(female_test_samples, male_test_samples)")

    test_dataset = (test_inputs, test_targets, test_mfccfiles)

    assert (len(training_inputs) == len(training_targets))

    # building up pybrain training dataset
    numberofInputs = len(training_inputs[0])

    startMemoryCounter()
    training_dataset = ClassificationDataSet(numberofInputs, nb_classes=2, class_labels=['Female', 'Male'])
    printMemoryDiffFromNow("ClassificationDataSet(numberofInputs, nb_classes=2, class_labels=['Female', 'Male'])")

    startMemoryCounter()
    for samplenum in xrange(0, len(training_inputs)):

        training_dataset.addSample(training_inputs[samplenum], [training_targets[samplenum]]) #TODO: this is the core problem for memory lost, i dont want to lose addSample and I'm not sure this adds a link instead of an instance --> answer I belive: it adds an instance

        if (samplenum%50000==0):
            printMemoryDiffFromNow("training_dataset.addSample... sampleNum: "+str(samplenum))
            startMemoryCounter()

    #
    # print ''
    # deep_getsizeof(training_dataset,'training_dataset',0)
    #

    printMemoryDiffFromNow("training_dataset.addSample... last sampleNum")

    if hiddenNeurons is None:
        hiddenNeurons = (training_dataset.indim + training_dataset.outdim) / 2

    """
		Computing results folder
	"""
    run_path = createRunFolder(resultsFolder=resultsFolder)

    input_params_file = os.path.join(run_path, 'inputParams.txt')
    input_params = {
        'learningRate': learningRate,
        'inputs': training_dataset.indim,
        'outputs': training_dataset.outdim,
        'hiddenNeurons': hiddenNeurons,
        'bias': bias,
        'momentum': momentum,
        'maxIterations': maxIterations,
        'femaleDataDir': femaleDataDir,
        'maleDataDir': maleDataDir,
        'datasetSize': len(training_dataset),
        'signalLength': signalLength,
        'signalSampleBuffer': signalSampleBuffer,
        'signalClass': signalClass.__name__,
        'resultsFolder': resultsFolder,
        'checkclassdir': checkclassdir
    }
    writeAsJson(input_params, input_params_file, indent=4)

    training_dataset_file = os.path.join(run_path, 'training_dataset.txt')
    writeAsJson(training_mfccfiles, training_dataset_file, indent=4)

    test_dataset_file = os.path.join(run_path, 'test_dataset.txt')
    test_mfccfiles = test_dataset[2]  # test_dataset format: (inputs,outputs,mfccfiles)
    writeAsJson(test_mfccfiles, test_dataset_file)

    startMemoryCounter()
    network = buildNetwork(training_dataset.indim, hiddenNeurons, hiddenNeurons/2, hiddenNeurons/3, training_dataset.outdim, bias=bias)
    printMemoryDiffFromNow("network = buildNetwork(...)")

    startMemoryCounter()
    trainer = BackpropTrainer(network, training_dataset, learningrate=learningRate, momentum=momentum, verbose=False)
    printMemoryDiffFromNow("trainer = BackpropTrainer(...)")

    epoch_error = 0  # keeps track of the last error
    tr_accuracy_male = 0
    tr_accuracy_female = 0
    tr_error = 0
    total_time = 0
    # training with training dataset
    startMemoryCounter()
    for epoch in xrange(0, maxIterations):
        print 'started train'
        timestampBeforeTrain = time.time();
        epoch_error = trainer.train()
        timestampAfterTrain = time.time();
        total_time = total_time + (timestampAfterTrain-timestampBeforeTrain)
        tr_accuracy_male, tr_accuracy_female = testOnDataset(training_dataset, network)
        tr_error_male = 1 - tr_accuracy_male
        tr_error_female = 1 - tr_accuracy_female

        printMemoryDiffFromNow("train iteration "+str(epoch)+ " Performance: "+str(timestampAfterTrain-timestampBeforeTrain))
        startMemoryCounter()


    pickleDumpObject(network,"tmpFiles/network_saved.p");

    training_error_male = tr_error_male
    training_accuracy_male = 1 - training_error_male
    training_error_female = tr_error_female
    training_accuracy_female = 1 - training_error_female

    avg_time_per_train = total_time/maxIterations

    test_results_file = os.path.join(run_path, 'test_results.txt')
    test_error = testOnCustomDataset(test_dataset, network, signalClass, test_results_file,performenceResult)
    test_accuracy = 1 - test_error

    performenceResult.append(training_accuracy_male)
    performenceResult.append(training_accuracy_female)

    print '----------------------------------------------------------------'
    print '**** Training Results:'
    print '----------------------------------------------------------------'
    print '* Training MSE : %s' % (epoch_error)
    print '* Training Dataset Male Accuracy: %s' % (training_accuracy_male)
    print '* Training Dataset Male Error: %s' % (training_error_male)
    print '* Training Dataset Female Accuracy: %s' % (training_accuracy_female)
    print '* Training Dataset Female Error: %s' % (training_error_female)
    print '* Test Dataset Accuracy: %s' % (test_accuracy)
    print '* Test Dataset Error %s' % (test_error)
    print '* Avg train(back propagation) action:  %s' % (avg_time_per_train)
    print '* Training dataset size:  %s' % len(training_inputs[0])
    print '* Test dataset size:  %s' % len(test_dataset[0])
    print '----------------------------------------------------------------'
    print 'Dumping Results in \"results_out.txt\" file '
    print '----------------------------------------x------------------------'

    """
		Dumping results in out file
	"""
    results_out_file = os.path.join(run_path, 'results_out.txt')
    results_out = {
        'MSE': epoch_error,
        'Training MSE:': (epoch_error),
        'Training Dataset Male Accuracy:':(training_accuracy_male),
        'Training Dataset Male Error:':(training_error_male),
        'Training Dataset Female Accuracy:':(training_accuracy_female),
        'Training Dataset Female Error:':(training_error_female),
        'Test Dataset Accuracy:':(test_accuracy),
        'Test Dataset Error:':(test_error),
        'Avg train(back propagation) action:':(avg_time_per_train),
        'Training dataset size:':len(training_inputs[0]),
        'Test dataset size:':len(test_dataset[0])
    }
    writeAsJson(results_out, results_out_file, indent=4)

    with open(test_results_file, 'a') as outfile:
        outfile.write(  '\n===========INPUT===========\n')
        outfile.write(  'processType: %s\n'%processType)
        outfile.write(  'maxIterations: %s\n' % maxIterations)
        outfile.write(  'signal buffer: %s\n' % signalSampleBuffer)
        outfile.write(  'signal length: %s\n' % signalLength)
        outfile.write( '\n===========Summary===========\n')
        outfile.write( '* Training MSE : %s\n' % (epoch_error))
        outfile.write( '* Training Dataset Male Accuracy: %s\n' % (training_accuracy_male))
        outfile.write( '* Training Dataset Male Error: %s\n' % (training_error_male))
        outfile.write( '* Training Dataset Female Accuracy: %s\n' % (training_accuracy_female))
        outfile.write( '* Training Dataset Female Error: %s\n' % (training_error_female))
        outfile.write( '* Test Dataset Accuracy: %s\n' % (test_accuracy))
        outfile.write( '* Test Dataset Error %s\n' % (test_error))
        outfile.write( '* Avg train(back propagation) action:  %s\n' % (avg_time_per_train))
        outfile.write( '* Training dataset size:  %s\n' % len(training_dataset))
        outfile.write( '* Test dataset size:  %s\n' % len(test_dataset[0]))

    if checkclassdir is not None:
        classification_out_filename = 'classification_out.txt'
        classification_out_file = os.path.join(run_path, classification_out_filename)
        """
		Classification for unlabeled samples
		"""
        print '----------------------------------------------------------------'
        print '**** Classifying samples in \"%s\" directory. ' % (checkclassdir)
        print '**   Dumping results in  \"%s\" file ' % (classification_out_filename)
        print '----------------------------------------------------------------'

        sample_files, classifications, activation_values = classifyUnlabeledSamples(checkclassdir, network, signalClass,
                                                                                    signalLength, signalSampleBuffer)
        assert ((len(sample_files) == len(classifications)) and (len(sample_files) == len(activation_values)))

        with open(classification_out_file, "a") as outfile:
            for samplenum, _ in enumerate(sample_files):
                classification_message = "#%s file \"%s\" classified as <%s>. Activation value: %s " % (
                samplenum + 1, sample_files[samplenum], classifications[samplenum], activation_values[samplenum])
                print classification_message
                outfile.write(classification_message + '\n')
    else:
        print 'No additional samples were specified. <checkclassdir>'
    """
		Dumping network in pickle file
	"""
    network_result_file = os.path.join(run_path, 'network.pickle')

    pickleDumpObject(network, network_result_file)

    return performenceResult

def main(args):
    arguments = extractCommandParams(args)

    # mandatory args
    learningRate = arguments["learningrate"]
    maxIterations = arguments["iterations"]
    femaleDataDir = arguments["femaledir"]
    maleDataDir = arguments["maledir"]
    processType = arguments["processType"]

    # optional args
    DEFAULT_MOMENTUM = 0.
    DEFAULT_BIAS = True
    DEFAULT_SIGNAL_LENGTH = 15
    DEFAULT_SIGNAL_CLASS = avgActivationValue
    DEFAULT_RESULTS_FOLDER = 'gender-class-runs'  # default name of folder where to place the result files
    DEFAULT_CHECK_CLASS_DIR = None
    DEFAULT_HIDDEN_NEURONS = None  # flag, if none the number is based on the input units
    DEFAULT_SIGNAL_SAMPLE_BUFFER = 1

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
        hiddenNeurons = arguments["hiddenneurons"]
    else:
        hiddenNeurons = DEFAULT_HIDDEN_NEURONS

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

    if "signalSampleBuffer" in arguments:
        signalSampleBuffer = arguments["signalSampleBuffer"]
    else:
        signalSampleBuffer = DEFAULT_SIGNAL_SAMPLE_BUFFER

    performenceResult = trainGenderClassification(learningRate=learningRate, hiddenNeurons=hiddenNeurons, bias=bias,
                              maxIterations=maxIterations, femaleDataDir=femaleDataDir,
                              maleDataDir=maleDataDir, momentum=momentum, signalLength=signalLength,
                              signalClass=signalClass,
                              resultsFolder=resultsFolder, checkclassdir=checkclassdir,signalSampleBuffer=signalSampleBuffer,processType=processType)

    return performenceResult

"""
    Main program
"""
if __name__ == '__main__':
    main(sys.argv[1:])