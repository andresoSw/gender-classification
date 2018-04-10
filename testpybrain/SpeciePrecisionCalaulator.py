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

                #PRECISION
                if (female_corrects+female_incorrects==0):
                    female_precision = 0
                else:
                    female_precision = female_corrects / float(female_corrects+female_incorrects)

                if (male_corrects+male_incorrects==0):
                    male_precision = 0
                else:
                    male_precision = male_corrects / float(male_corrects+male_incorrects)

                # RECALL
                if (female_corrects+male_incorrects==0):
                    female_recall = 0
                else:
                    female_recall = female_corrects / float(female_corrects+male_incorrects)

                if (male_corrects+female_incorrects==0):
                    male_recall = 0
                else:
                    male_recall = male_corrects / float(male_corrects+female_incorrects)

                printedStr ='Male precision: '+str(format(male_precision, '.2f'))+' '
                printedStr += 'Male recall: ' + str(format(male_recall, '.2f'))+' '
                printedStr += 'Female precision: ' + str(format(female_precision, '.2f'))+' '
                printedStr += 'Female recall : ' + str(format(female_recall, '.2f'))+', '
                printedStr += key

                output_file.write(printedStr+'\n');

        output_file.close()