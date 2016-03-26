import sys, getopt
import os
import json
import pickle

"""
   Utilities module
"""

def writeAsJson(content,file,indent=0):
   with open(file, 'w') as outfile:
      json.dump(content, outfile,indent=indent)

def pickleDumpObject(obj,_file):
   pickle.dump(obj, file(_file,'wb'))

def pickleLoadObject(_file):
   obj = pickle.load(file(_file,'rb'))
   return obj

"""
   @return a dictionary of parsed arguments
"""
def extractCommandParams(argv):
   how_to_use_message = '$ Usage: \n\tShort ARGS: testtraining.py -l <rate> ' \
                        '-h <howmany> -b <true,false> -i <howmany> -f <path> -m <path>\n'\
                        '\tLong  ARGS: testtraining.py --learningrate <rate> ' \
                        '--hiddenneurons <howmany> --bias <true,false> --iterations <howmany>'\
                        '--femaledir <path> --maledir <path>\n\n'\
                        '\t[OPTIONAL ARGS] --signallength <length> --signalcount <count> --checkclassdir <parh> --rfolder <path> '\
                        '--elitism <true,false> --pselection <roulette,rank,tournament>\n\n'\
                        '$ Please refer to the README.md file for further explanation\n'

   mandatory_args = [("-l","--learningrate"),("-h","--hiddenneurons"),("-b","--bias"),("-i","--iterations"),("-f","--femaledir"),("-m","--maledir")]
   optional_args = [("--signallength"),("--signalcount"),("--checkclassdir"),("--rfolder")]

   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'l:h:b:i:f:m:',['learningrate=','hiddenneurons=','bias=','iterations=',
      												'femaledir=','maledir=','signallength=','signalcount=','checkclassdir=',
      												'rfolder='])
   except getopt.GetoptError:
      print how_to_use_message
      sys.exit(2)

   parsed_arguments = {}

   for opt, arg in opts:
      #mandatory args
      if opt in ("-l","--learningrate"):
         parsed_arguments["learningrate"] = float(arg)
      elif opt in ("-h","--hiddenneurons"):
         parsed_arguments["hiddenneurons"] = int(arg)
      elif opt in ("-b","--bias"):
         parsed_arguments["bias"] = bool(arg)
      elif opt in ("-i","--iterations"):
         parsed_arguments["iterations"] = int(arg)
      elif opt in ("-f","--femaledir"):
         parsed_arguments["femaledir"] = arg
      elif opt in ("-m","--maledir"):
         parsed_arguments["maledir"] = arg

      #optional args
      elif opt == "--signallength":
         parsed_arguments["signallength"] = int(arg)
      elif opt == "--signalcount":
         parsed_arguments["signalcount"] = int(arg)
      elif opt == "--checkclassdir":
      	parsed_arguments["checkclassdir"] = arg
      elif opt == "--rfolder":
      	parsed_arguments["rfolder"] = arg

   return parsed_arguments


"""
   Creates a folder where to dump run results. Runs have the same prefix and are identified
   by an index starting at 0.
   @param resultsFolder the parent folder
   @param runFolderPrefix a prefix to be used to name run folders
   @return the path pointing to the created directory
"""
def createRunFolder(resultsFolder,runFolderPrefix='run_'):
      """
         Computing run folder
      """
      found = False
      dir_index = 0
      dir_prefix = os.path.join(resultsFolder,runFolderPrefix)

      #find an unused index-based name for the new run folder
      while not found:
         run_path = dir_prefix+str(dir_index)
         found =  not os.path.isdir(run_path)
         dir_index += 1

      #create the directory
      try: 
          os.makedirs(run_path)
      except OSError:
          if not os.path.isdir(run_path):
              raise
      return run_path

if __name__ == "__main__":
   #ignoring the name of the program from the command line args
   arguments = extractCommandParams(sys.argv[1:]) 

   #mandatory args
   learningRate = arguments["learningrate"]
   hiddenNeurons = arguments["hiddenneurons"]
   bias = arguments["bias"]
   maxIterations = arguments["iterations"]
   female_data_dir = arguments["femaledir"]
   male_data_dir = arguments["maledir"]

   #optional args
   DEFAULT_SIGNAL_LENGTH = 15
   DEFAULT_SIGNAL_COUNT = 1
   DEFAULT_RESULTS_FOLDER = 'gender-class-runs' #default name of folder where to place the result files

   if "signallength" in arguments:
      signal_length = arguments["signallength"]
   else:
      signal_length = DEFAULT_SIGNAL_LENGTH

   if "signalcount" in arguments:
      signal_count = arguments["signalcount"]
   else:
      signal_count = DEFAULT_SIGNAL_COUNT

   if "rfolder" in arguments:
      results_folder = arguments["rfolder"]
   else:
      results_folder = DEFAULT_RESULTS_FOLDER 

   print arguments