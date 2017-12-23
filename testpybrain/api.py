from flask import Flask, render_template, request
from os import listdir
from os.path import isfile, join
import pickle
import json
import jsonpickle
import sqlite3
import testtraining

app = Flask(__name__)

# GENDER_ENGINE_PATH = "/home/shahar963/PycharmProjects/localTestRun"
# DB_PATH = '/home/shahar963/PycharmProjects/trainAPI/mydb'
#
def connectToDB(path):
    db = sqlite3.connect(path)
    return db

def closeDB(db):
    db.commit;
    db.close

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
#     with open('network_saved.p', 'r') as input_file:
#         content = input_file.read()
#     blob = sqlite3.Binary(content)
#
#     new_network = ("test network",0.01,100,320,20,'mfcc',blob);
#
#     cur.execute(sql,new_network);
#
#     db.commit();
#
#     db.close()
#     return "Hello World!"
#
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

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        b=f.save(f.filename)
        prediction = testtraining.testFileOnDefaultNetwork(f.filename,testtraining.avgActivationValue)

        return prediction

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=42426,debug=True)