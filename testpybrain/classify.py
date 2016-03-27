from utilities import pickleDumpObject,pickleLoadObject
from testtraining import getVoiceSignal,getGender
import sys, getopt
import os
from os.path import isfile,join
import pickle
import numpy as np

def getCommandParams(argv):
   how_to_use_message = '$ Usage: \n\tShort ARGS: classify.py -m <mfccfile> ' \
                        ' -n <networkfile> \n'\
                        '\tLong  ARGS: classify.py --mfcc <mfccfile> ' \
                        ' --network <networkfile>\n\n'
   mandatory_args = [("-m","--mfcc"),("-n","--network")]

   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'m:n:',['mfcc=','network='])
   except getopt.GetoptError:
      print how_to_use_message
      sys.exit(2)

   parsed_arguments = {}

   for opt, arg in opts:
      #mandatory args
      if opt in ("-m","--mfcc"):
         parsed_arguments["mfcc"] = arg
      elif opt in ("-n","--network"):
         parsed_arguments["network"] = arg
   return parsed_arguments


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

def classifyUnlabeledSample(mfccfile,network):
	sample_files = None
	classifications = None
	activationValue = None

	if (not isfile(mfccfile)) or ('.mfcc' not in mfccfile): 
		return None

	file_data = np.loadtxt(mfccfile)
	vs = getVoiceSignal(file_data,network.signalLength,network.signalCount)
	result = avgActivationValue(vs,network)
	
	classification = getGender(result)

	return classification,result

def forceClassification(activationValue,femaleCut=0.2,maleCut=0.8):
	if activationValue <= femaleCut:
		return 'female'
	if activationValue >= maleCut:
		return 'male'
	return 'unknown' 

if __name__ == "__main__":
	arguments = getCommandParams(sys.argv[1:]) 
	mfccfile = arguments["mfcc"]
	network_result_file= arguments["network"]
	network = pickleLoadObject(network_result_file)
	classification,activationValue = classifyUnlabeledSample(mfccfile,network)
	print forceClassification(activationValue)