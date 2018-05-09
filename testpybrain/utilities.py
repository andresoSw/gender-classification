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
                        ' -i <howmany> -f <path> -m <path> -p <processType>\n'\
                        '\tLong  ARGS: testtraining.py --learningrate <rate> ' \
                        ' --iterations <howmany> --processType <mfcc, custom_mfcc,mfcc_without_dct or wav>'\
                        '--femaledir <path> --maledir <path>\n\n'\
                        '\t[OPTIONAL ARGS] --hiddenneurons <howmany> --momentum <rate> --bias <true,false> --signallength <length>'\
                        '--signalclass <avg or mode> --checkclassdir <parh> --rfolder <path> '\
                        '$ Please refer to the README.md file for further explanation\n'

   mandatory_args = [("-l","--learningrate"),("-i","--iterations"),("-f","--femaledir"),("-m","--maledir"),("-p","--processType")]
   optional_args = [("--hiddenneurons"),("--momentum"),("--bias"),("--signallength"),("--signalclass"),("--checkclassdir"),("--rfolder"),("--signalSampleBuffer")]

   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'l:i:f:m:p:',['learningrate=','iterations=',
      												'femaledir=','maledir=','hiddenneurons=','momentum=',
                                          'bias=','signallength=','signalclass=','checkclassdir=',
      												'rfolder=','signalSampleBuffer='])
   except getopt.GetoptError:
      print how_to_use_message
      sys.exit(2)

   parsed_arguments = {}

   for opt, arg in opts:
      #mandatory args
      if opt in ("-l","--learningrate"):
         parsed_arguments["learningrate"] = float(arg)
      elif opt in ("-i","--iterations"):
         parsed_arguments["iterations"] = int(arg)
      elif opt in ("-f","--femaledir"):
         parsed_arguments["femaledir"] = arg
      elif opt in ("-m","--maledir"):
         parsed_arguments["maledir"] = arg
      elif opt in ("-p","--processType"):
          parsed_arguments["processType"] = arg

      #optional args
      elif opt == "--signallength":
         parsed_arguments["signallength"] = int(arg)
      elif opt == "--checkclassdir":
      	parsed_arguments["checkclassdir"] = arg
      elif opt == "--rfolder":
      	parsed_arguments["rfolder"] = arg
      elif opt == ("--hiddenneurons"):
         parsed_arguments["hiddenneurons"] = int(arg)
      elif opt == "--momentum":
         parsed_arguments["momentum"] = float(arg)
      elif opt == "--bias":
         parsed_arguments["bias"] = arg.lower() in ['true','t']
      elif opt == "--signalclass":
         if not arg in ("avg","mode"):
            print '* Warning: Invalid argument for signal class \"%s\", expected \"avg\" or \"mode\"\nAvg will be used by default' %(arg)
            parsed_arguments["signalclass"] = "avg"
         else:
            parsed_arguments["signalclass"] = arg
      elif opt == "--signalSampleBuffer":
          parsed_arguments["signalSampleBuffer"] = int(arg)


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