from flask import Flask, request
import json
import sqlite3
import testtraining
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def connectToDB(path):
    db = sqlite3.connect(path)
    return db

@app.route('/', methods=['GET'])
def isAlive():
    return "Hello world"

@app.route('/deleteNetwork', methods=['GET'])
def deleteNetwork():
    networkName = str(request.args['name'])

    db = connectToDB("mydb")
    cur = db.cursor()

    cur.execute('DELETE FROM TRAINED_NEURAL_NETWORKS WHERE description="'+networkName+'";')

    db.commit()
    db.close()

    return networkName

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