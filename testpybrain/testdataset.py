from utilities import pickleLoadObject,createRunFolder,writeAsJson
from testtraining import getVoiceSignal,getGender,modeActivationValue,avgActivationValue,getData,testOnCustomDataset,combineSamples
import sys, getopt
import os

def getCommandParams(argv):
   how_to_use_message = '$ Usage: \n\tShort ARGS: testdataset.py -n <networkfile> -m <path> ' \
                        ' -f <path>\n'\
                        '\tLong  ARGS: classify.py --network <networkfile> ' \
                        ' --maledir <path> --female <path> \n\n'\
                        '\t[OPTIONAL ARGS] --signalclass <avg or mode>  --rfolder <path> --signalcount <count>\n'

   
   mandatory_args = [("-n","--network"),("-m","--male"),("-f","--female")]
   optional_args = [("--signalclass"),("--signalcount"),("--rfolder")]


   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'n:m:f:',['network=','maledir=','femaledir=','signalclass=','signalcount=','rfolder='])
   except getopt.GetoptError:
      print how_to_use_message
      sys.exit(2)

   parsed_arguments = {}

   for opt, arg in opts:
      #mandatory args
      if opt in ("-n","--network"):
         parsed_arguments["network"] = arg
      elif opt in ("-f","--femaledir"):
         parsed_arguments["femaledir"] = arg
      elif opt in ("-m","--maledir"):
         parsed_arguments["maledir"] = arg

      #optional args
      elif opt == "--rfolder":
         parsed_arguments["rfolder"] = arg
      elif opt == "--signalclass":
         if not arg in ("avg","mode"):
            print '* Warning: Invalid argument for signal class \"%s\", expected \"avg\" or \"mode\"\nAvg will be used by default' %(arg)
            parsed_arguments["signalclass"] = "avg"
         else:
            parsed_arguments["signalclass"] = arg
      elif opt == "--signalcount":
         parsed_arguments["signalcount"] = int(arg)
   return parsed_arguments

"""
   Main Function
"""
def testGenderClassification(network,networkFile,maleDataDir,femaleDataDir,signalClass,signalCount,resultsFolder):
   print '----------------------------------------------------------------'
   print '***** Running FeedForward Test with parameters:\n'
   print '* networkfile    : %s' %(networkFile)
   print '* signalCount    : %s' %(signalCount)
   print '* signalClass    : %s' %(signalClass.__name__)
   print '* maleDataDir    : %s' %(maleDataDir)
   print '* femaleDataDir  : %s' %(femaleDataDir)
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
      'networkfile':networkFile,
      'signalCount':signalCount,
      'signalClass':signalClass.__name__,
      'maleDataDir':maleDataDir,
      'femaleDataDir':femaleDataDir,
      'resultsFolder':resultsFolder 
   }
   writeAsJson(input_params,input_params_file,indent=4)

   #extracting female and male samples. All samples will be used for testing
   _,female_test_samples = getData(femaleDataDir,network.signalLength,network.signalCount,testProportion=1.0)
   _,male_test_samples = getData(maleDataDir,network.signalLength,signalCount,testProportion=1.0)

   test_inputs,test_targets,test_mfccfiles = combineSamples(female_test_samples,male_test_samples)
   test_dataset = (test_inputs,test_targets,test_mfccfiles)   

   test_results_file = os.path.join(run_path,'test_results.txt')
   test_error = testOnCustomDataset(test_dataset,network,signalClass,test_results_file,[])

   test_accuracy = 1-test_error

   print '----------------------------------------------------------------'
   print '**** Dataset Test Results:'
   print '----------------------------------------------------------------'
   print '* Test Dataset Accuracy: %s' %(test_accuracy)
   print '* Test Dataset Error %s' %(test_error)
   print '----------------------------------------------------------------'
   print 'Dumping Results in \"test_results.txt\" file '
   print '----------------------------------------------------------------'


if __name__ == "__main__":
   arguments = getCommandParams(sys.argv[1:]) 

   #mandatory args
   network_result_file= arguments["network"]
   femaleDataDir = arguments["femaledir"]
   maleDataDir = arguments["maledir"]

   network = pickleLoadObject(network_result_file)

   #optional args
   DEFAULT_SIGNAL_CLASS = avgActivationValue
   DEFAULT_SIGNAL_COUNT = 1
   DEFAULT_RESULTS_FOLDER = 'test-dataset-runs' #default name of folder where to place the result files
   
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

   if "rfolder" in arguments:
     resultsFolder = arguments["rfolder"]
   else:
     resultsFolder = DEFAULT_RESULTS_FOLDER

   testGenderClassification(network=network,networkFile=network_result_file,maleDataDir=maleDataDir,femaleDataDir=femaleDataDir,
                           signalClass=signalClass,signalCount=signalCount,resultsFolder=resultsFolder)
   