from utilities import pickleDumpObject,pickleLoadObject
from testtraining import getVoiceSignal,getGender,modeActivationValue,avgActivationValue
import sys, getopt
import os
from os.path import isfile,join
import pickle
import numpy as np
from collections import Counter

def getCommandParams(argv):
   how_to_use_message = '$ Usage: \n\tShort ARGS: classify.py -m <mfccfile> ' \
                        ' -n <networkfile> -s <avg or mode>\n'\
                        '\tLong  ARGS: classify.py --mfcc <mfccfile> ' \
                        ' --network <networkfile> \n\n'\
						'\t[OPTIONAL ARGS] --signalclass <avg or mode> --signalcount <count>\n'

   mandatory_args = [("-m","--mfcc"),("-n","--network")]
   optional_args = [("--signalclass"),("--signalcount")]

   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'m:n:',['mfcc=','network=','signalclass=','signalcount='])
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
      elif opt == "--signalcount":
         parsed_arguments["signalcount"] = int(arg)
      elif opt in ("-s","--signalclass"):
      	if not arg in ("avg","mode"):
            print '* Warning: Invalid argument for signal class \"%s\", expected \"avg\" or \"mode\"\nAvg will be used by default' %(arg)
            parsed_arguments["signalclass"] = "avg"
        else:
            parsed_arguments["signalclass"] = arg
   return parsed_arguments

def classifyUnlabeledSample(mfccfile,network,signalClass,signalLength,signalCount):
	sample_files = None
	classifications = None
	activationValue = None

	if (not isfile(mfccfile)) or ('.mfcc' not in mfccfile): 
		return None

	file_data = np.loadtxt(mfccfile)
	vs = getVoiceSignal(file_data,signalLength,signalCount)
	result = signalClass(vs,network)
	
	classification = getGender(result)

	return classification,result

def forceClassification(activationValue,femaleCut=0.2,maleCut=0.8):
	if activationValue <= femaleCut:
		return 'female'
	if activationValue >= maleCut:
		return 'male'
	return 'unknown' 

"""
	Main function
"""
def classifySample(mfccfile,network,signalClass,signalLength,signalCount):
	classification,activationValue = classifyUnlabeledSample(mfccfile,network,signalClass,signalLength,signalCount)
	return forceClassification(activationValue)

if __name__ == "__main__":
	arguments = getCommandParams(sys.argv[1:]) 
	mfccfile = arguments["mfcc"]
	network_result_file= arguments["network"]
	network = pickleLoadObject(network_result_file)

	#optional args
	DEFAULT_SIGNAL_CLASS = avgActivationValue
	DEFAULT_SIGNAL_COUNT = network.signalCount


	if "signalclass" in arguments:
	  if arguments["signalclass"] == "mode":
	     signalClass = modeActivationValue
	  elif arguments["signalclass"] == "avg":
	     signalClass = avgActivationValue
	else:
	  signalClass = DEFAULT_SIGNAL_CLASS

	if "signalcount" in arguments:
		signalCount = arguments["signalcount"]
	else:
		signalCount = DEFAULT_SIGNAL_COUNT

	print classifySample(mfccfile,network,signalClass,network.signalLength,signalCount)
