from flask import Flask, render_template, request
from os import listdir
from os.path import isfile, join
import json
import jsonpickle
import sqlite3
import testtraining
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def connectToDB(path):
    db = sqlite3.connect(path)
    return db

# @app.route("/")
# def hello():
#     db = connectToDB("mydb")
#     cur = db.cursor()
#
#     sql = ''' INSERT INTO TRAINED_NEURAL_NETWORKS(  description,
#                                                     learningRate,
#                                                     maxIterations,
#                                                     signal_length,
#                                                     signal_sample_buffer,
#                                                     process_type,
#                                                     network)
#                   VALUES(?,?,?,?,?,?,?) '''
#
#     with open('tmpFiles/network_saved.p', 'r') as input_file:
#         content = input_file.read()
#     blob = sqlite3.Binary(content)
#
#     new_network = ("test network2",0.01,100,320,20,'mfcc',blob);
#
#     cur.execute(sql,new_network);
#
#     db.commit();
#
#     db.close()
#     return "Hello World!"

# @app.route("/testFile", methods=["POST"])
# def testFile():
#     print("stepped here")
#
#     return "output?"
#
# @app.route("/networks", methods=["GET"])
# def getNetworks():
#     db = connectToDB("mydb")
#     cur = db.cursor()
#
#     cur.execute('''SELECT   id,
#                             description,
#                             learningRate,
#                             maxIterations,
#                             signal_length,
#                             signal_sample_buffer,
#                             process_type
#                                 FROM TRAINED_NEURAL_NETWORKS''')
#     rows = cur.fetchall();
#
#     closeDB(db)
#
#     rows_JSON = json.dumps(rows)
#
#     return rows_JSON

@app.route('/trainNewNetwork', methods=['GET', 'POST'])
def trainNewNetwork():

    data = json.loads(request.data)

    description=str(data['name'])
    learningRate=str(data['learningRate'])
    maxIterations=str(data['iterations'])
    signalLength=str(data['signalLength'])
    signalSampleBuffer=str(data['signalSampleBuffer'])
    processType = str(data['processType'])

    res = testtraining.trainNewNetwork(description,learningRate,maxIterations,processType,signalLength,signalSampleBuffer)

    return "stub return value of trainNewNetwork function"+res

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('tmpFiles/'+f.filename)
        networkSelected = request.form['networkSelected'];

        prediction = testtraining.testFileOnNetwork(f.filename,networkSelected,testtraining.avgActivationValue)

        return prediction

@app.route('/getTrainedNetworks', methods=['GET'])
def getTrainedNetworks():
    db = connectToDB("mydb")
    cur = db.cursor()

    cur.execute('''SELECT   id,
                            description,
                            learningRate,
                            maxIterations,
                            signal_length,
                            signal_sample_buffer,
                            process_type,
                            male_training_precision,
                            female_training_precision,
                            male_test_precision,
                            male_test_recall,
                            female_test_precision,
                            female_test_recall
                                FROM TRAINED_NEURAL_NETWORKS''')
    rows = cur.fetchall();
    db.close()

    rowsJson=json.dumps(rows)

    return rowsJson

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=42426,debug=True)