# gender-classification
Gender classification from speech using neural networks

#### Installation
TODO

```bash
$ TODO
```
#### Execution
Execute gender classification project by running the following command

```bash
$ python testtraining.py <MANDATORY ARGS> <OPTIONAL ARGS>
```

Below there is a list of the mandatory and optional arguments to be provided respectively:

* Mandatory Arguments

| Argument                        |Short Version        | Long Version             | Expected Value                  |
|---------------------------------|---------------------|--------------------------|---------------------------------|
| Learning Rate                   |     -l              |    --learningrate        |          float number           |
| Number of Iterations            |     -i              |    --iterations          |          int number             |
| Female Samples Path             |     -f              |    --femaledir           |  path to female samples folder  |
| Male Samples Path               |     -m              |    --maledir             |  path to male samples folder    |

* Optional Arguments

| Argument                                 | Specification        |Expected Value        |Default Value                   |
|------------------------------------------|--------------------- |----------------------|--------------------------------|
| Number of Hidden Neurons                 |    --hiddenneurons   |          int number  | (inputUnits + outputUnits)/2   |
| Momentum                                 |--momentum            | float number         |   0.0                          |
| Bias                                     |--bias                | true or false        |   true                         |
| Signal Length                            |--signallength        | int number           |   15                           |
| Signal Count                             |--signalcount         | int number           |   1                            |
| Results Folder                           |--rfolder             | path to folder       |  /gender-classification-runs   |
| Unlabeled samples Path                   |--checkclassdir       | path to samples      |  None                          |


In case that any of the optional argument is not specified, its default value will be used instead.
Notes:
  * Number of input units comes from the number of mfcc coefficients taken into account for each sample.
  * There is only 1 output unit (2 classes: 'male' and 'female')

####Example of project invocation:

```bash
$ python testtraining.py -learningrate 0.01 -h 100 -b true -i 100 -f female -m male --rfolder my-classification-results
```
Note that short argument names and long argument names can be used indifferently


####Results description TODO
The following files will be created inside the result folders
* inputParams.txt
* classification_out.txt
* network.pickle
* results_out.txt

A description of the content of each file is summarized in the following table

|       Filename            |             Content Description                           |        Format          |
|---------------------------|-----------------------------------------------------------|------------------------|
| input_params.txt          | A summary of the input parameters provided by the user    | json                   |
| classification_out.txt    | The classification for the unlabeled samples, if specified| verbose                |
| network.pickle            | The serialized neural network                             | pickle                 |
| results_out.txt           | A summary of the test and training accuracy and errors    | json                   |

