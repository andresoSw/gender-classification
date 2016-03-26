# gender-classification
Gender classification from speech using neural networks


# GABIL-GA
Implementation of Genetic Algorithm (GABIL) for classification

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
| Number of Hidden Neurons        |     -h              |    --hiddenneurons       |          int number             |
| Bias                            |     -b              |    --bias                |          true or false          |
| Number of Iterations            |     -i              |    --iterations          |          int number             |
| Female Samples Path             |     -f              |    --femaledir           |  path to female samples folder  |
| Male Samples Path               |     -m              |    --maledir             |  path to male samples folder    |
,,("--checkclassdir"),("--rfolder")]

* Optional Arguments

| Argument                                 | Specification        |Expected Value        |Default Value                   |
|------------------------------------------|--------------------- |----------------------|--------------------------------|
| Signal Length                            |--signallength        | int number           |   15                           |
| Signal Count                             |--signalcount         | int number           |   1                            |
| Results Folder                           |--rfolder             | path to folder       |  /gender-classification-runs   |


In case that any of the optional argument is not specified, its default value will be used instead

####Example of project invocation:

```bash
$ python testtraining.py -learningrate 0.01 -h 100 -b true -i 2 -f female -m male --rfolder my-classification-results
```
Note that short argument names and long argument names can be used indifferently


####Results description TODO
The following files will be created inside the result folders
* input_params.txt

A description of the content of each file is summarized in the following table

|       Filename            |             Content Description                           |        Format          |
|---------------------------|-----------------------------------------------------------|------------------------|
| input_params.txt          | A summary of the input parameters provided by the user    | json                   |

