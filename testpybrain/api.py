from flask import Flask
from flask import request
from os import listdir
from os.path import isfile, join
import pickle
import json
import jsonpickle
import sqlite3
import testtraining

app = Flask(__name__)

GENDER_ENGINE_PATH = "/home/shahar963/PycharmProjects/localTestRun"
DB_PATH = '/home/shahar963/PycharmProjects/trainAPI/mydb'

def connectToDB():
    db = sqlite3.connect(DB_PATH)
    return db

def closeDB(db):
    db.commit;
    db.close

@app.route("/")
def hello():
    #
    # sql = ''' INSERT INTO TRAINED_NEURAL_NETWORKS(  description,
    #                                                 learningRate,
    #                                                 maxIterations,
    #                                                 signal_length,
    #                                                 signal_sample_buffer,
    #                                                 process_type,
    #                                                 network)
    #               VALUES(?,?,?,?,?,?,?) '''
    #
    # networksPath = GENDER_ENGINE_PATH + '/trained networks/'
    # networkFiles = [f for f in listdir(networksPath) if isfile(join(networksPath, f))]
    # fileHandle = open(networkFiles[0],'r');
    # content = fileHandle.read()
    # blob = sqlite3.Binary(content)
    #
    # new_network = ("test network",0.01,100,320,20,'mfcc',blob);
    #
    # cursor.execute(sql,new_network);
    #
    # db.commit();
    #
    # db.close()
    return "Hello World!"

@app.route("/testFile", methods=["GET"])
def getNetworks():
    db = connectToDB()
    cur = db.cursor()

    cur.execute('''SELECT   id,
                            description,
                            learningRate,
                            maxIterations,
                            signal_length,
                            signal_sample_buffer,
                            process_type
                                FROM TRAINED_NEURAL_NETWORKS''')
    rows = cur.fetchall();

    closeDB(db)

    rows_JSON = json.dumps(rows)

    return rows_JSON

@app.route("/networks", methods=["GET"])
def getNetworks():
    db = connectToDB()
    cur = db.cursor()

    cur.execute('''SELECT   id,
                            description,
                            learningRate,
                            maxIterations,
                            signal_length,
                            signal_sample_buffer,
                            process_type
                                FROM TRAINED_NEURAL_NETWORKS''')
    rows = cur.fetchall();

    closeDB(db)

    rows_JSON = json.dumps(rows)

    return rows_JSON

if __name__ == '__main__':
    app.run(debug=True)