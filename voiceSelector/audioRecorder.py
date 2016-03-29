# -*- coding: utf-8 -*-
"""
    SOURCE FROM https://gist.github.com/sloria/5693955
"""
from recorder import Recorder
from utilities import pickleDumpObject,pickleLoadObject
from mfccGenerator import createMFCC
from classify import classifySample,classifyUnlabeledSample
from testtraining import modeActivationValue,avgActivationValue,getGender
import sys, getopt
import os

def getCommandParams(argv):
   how_to_use_message = '$ Usage: \n\tShort ARGS: audioRecorder.py -s <howmany> ' \
                        ' -n <networkfile>\n'\
                        '\tLong  ARGS: classify.py --seconds <howmany> ' \
                        ' --network <networkfile> \n\n'\
						'\t[OPTIONAL ARGS] --signalclass <avg or mode> --signalcount <count>\n'

   mandatory_args = [("-s","--seconds"),("-n","--network")]
   optional_args = [("--signalclass"),("--signalcount")]

   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'s:n:',['seconds=','network=','signalclass=','signalcount='])
   except getopt.GetoptError:
      print how_to_use_message
      sys.exit(2)

   parsed_arguments = {}

   for opt, arg in opts:
      #mandatory args
      if opt in ("-s","--seconds"):
         parsed_arguments["seconds"] = float(arg)
      elif opt in ("-n","--network"):
         parsed_arguments["network"] = arg
      elif opt == "--signalcount":
         parsed_arguments["signalcount"] = int(arg)
      elif opt == ("--signalclass"):
      	if not arg in ("avg","mode"):
            print '* Warning: Invalid argument for signal class \"%s\", expected \"avg\" or \"mode\"\nAvg will be used by default' %(arg)
            parsed_arguments["signalclass"] = "avg"
        else:
            parsed_arguments["signalclass"] = arg
   return parsed_arguments


def getCertaintyMessage(activationValue,femaleCut=0.35,maleCut=0.65,certainMessage="I'm certain",uncertainMessage="I think"):
	if ((activationValue <= femaleCut) or (activationValue >= maleCut)):
		return certainMessage
	elif ((activationValue >= femaleCut) and (activationValue <= maleCut)):
		return uncertainMessage

if __name__ == "__main__":
	arguments = getCommandParams(sys.argv[1:]) 
	#mandatory args
	network_result_file= arguments['network']
	network = pickleLoadObject(network_result_file)

	record_duration = arguments['seconds'] #seconds

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

	wav_file_name = 'sample.wav'
	rec = Recorder(channels=2)
	print '===================================='
	print '********** Start Recording'
	print '********** Recording . . .'
	with rec.open(wav_file_name, 'wb') as recordfile:
	    recordfile.record(duration=record_duration)
	print '===================================='
	print '********** Stop Recording'
	print '===================================='
	print '********** Voice Signal Results:'
	mfcc_file = createMFCC(wav_file_name)
	classification,activationValue = classifyUnlabeledSample(mfcc_file,network,signalClass,network.signalLength,signalCount)
	certainty = getCertaintyMessage(activationValue)
	print 'According to your voice, %s you are a %s!' %(certainty,getGender(activationValue))
