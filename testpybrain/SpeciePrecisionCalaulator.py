import csv
import re

TEST_RESULTS_FILE_PATH = "resources/test_results.txt"
SPECIES_PRECISION_CLASSIFICATION_PATH="output/species_precision_classification.txt"
MAPPING_CSV_PATH = "resources/CatalogNum-Specie mapping.csv"

def calculateSpeciePrecisionAndRecall(RESULTS_FILE_PATH,SPECIES_CLASSIFICATION_PATH,MAPPING_CSV):
    catalogNumArr = []
    speciesArr = []
    dict = {}

    with open(MAPPING_CSV, 'r') as csvfile:
        mappingReader = csv.reader(csvfile)
        for row in mappingReader:
            catalogNumArr.append(row[0])
            speciesArr.append(row[1])

    with open(RESULTS_FILE_PATH) as f:
        content = f.readlines()

        for line in content:
            if  "================================" in line:
                break

            lineCatalogNum = re.search('/(.+)_',line).group(1)
            lineSpecie = speciesArr[catalogNumArr.index(lineCatalogNum)]
            gender = re.search(' (.+)/', line).group(1)
            correct = not ("INCORRECT" in line)

            # TODO: It is possible to go to in a higher resolution into specific tweet
            if(not lineSpecie in dict):
                dict[lineSpecie] ={ 'male':{'True':0,'False':0}, 'female':{'True':0,'False':0}}

            dict[lineSpecie][gender][str(correct)] += 1

        with open(SPECIES_CLASSIFICATION_PATH, "w+") as output_file:
            for key, value in dict.items():
                male_corrects = value['male']['True']
                male_incorrects = value['male']['False']
                female_corrects = value['female']['True']
                female_incorrects = value['female']['False']

                # PRECISION
                if (female_corrects + female_incorrects == 0):
                    female_precision = "Non "
                else:
                    female_precision = female_corrects / float(female_corrects + female_incorrects)

                if (male_corrects + male_incorrects == 0):
                    male_precision = "Non "
                else:
                    male_precision = male_corrects / float(male_corrects + male_incorrects)

                # RECALL
                if (female_corrects + male_incorrects == 0):
                    female_recall = "Non "
                else:
                    female_recall = female_corrects / float(female_corrects + male_incorrects)

                if (male_corrects + female_incorrects == 0):
                    male_recall = "Non "
                else:
                    male_recall = male_corrects / float(male_corrects + female_incorrects)

                if (type(male_precision) is float):
                    male_precision = str(format(male_precision, '.2f'))
                printedStr = 'Male precision: ' + male_precision + ' '

                if (type(male_recall) is float):
                    male_recall = str(format(male_recall, '.2f'))
                printedStr += 'Male recall: ' + male_recall + ' '

                if (type(female_precision) is float):
                    female_precision = str(format(female_precision, '.2f'))
                printedStr += 'Female precision: ' + female_precision + ' '

                if (type(female_recall) is float):
                    female_recall = str(format(female_recall, '.2f'))
                printedStr += 'Female recall: ' + female_recall + ' '

                printedStr += key

                output_file.write(printedStr + '\n');

        output_file.close()

def calculateTweetPrecision(RESULTS_FILE_PATH, TWEETS_CLASSICIATION_PATH, MAPPING_CSV):
    catalogNumArr = []
    speciesArr = []
    dict = {}

    with open(MAPPING_CSV, 'r') as csvfile:
        mappingReader = csv.reader(csvfile)
        for row in mappingReader:
            catalogNumArr.append(row[0])
            speciesArr.append(row[1])

    with open(RESULTS_FILE_PATH) as f:
        content = f.readlines()

        for line in content:
            if "================================" in line:
                break

            lineCatalogNum = re.search('/(.+)_', line).group(1)
            lineSpecie = speciesArr[catalogNumArr.index(lineCatalogNum)]
            correct = not ("INCORRECT" in line)

            if (not lineSpecie in dict):
                dict[lineSpecie] = {}

            if (not lineCatalogNum in dict[lineSpecie]):
                dict[lineSpecie][lineCatalogNum] = {'True': 0, 'False': 0}

            dict[lineSpecie][lineCatalogNum][str(correct)] += 1

        with open(TWEETS_CLASSICIATION_PATH, "w+") as output_file:
            for key, specie in dict.items():

                output_file.write(str(key) + ':\n')
                for key2, CorrectMapping in specie.items():

                    positive = float(CorrectMapping['True'])
                    negative = float(CorrectMapping['False'])
                    corrects = str(int(positive/(positive+negative))*100)

                    printedStr = str(key2)+' '

                    printedStr += str(corrects)+'%'

                    output_file.write(printedStr + '\n')

                output_file.write('\n')

        output_file.close()