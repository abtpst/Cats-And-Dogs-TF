'''
Created on Nov 27, 2017

@author: abhijit.tomar
'''

import os, sys
sys.path.append(os.getcwd() + "/src")
import tflearn
from flask import Flask, render_template, request, jsonify
from storage.SQLliteStorage import CNNSQL
from data_process import Prepare
from neurals.Networks import CatsAndDogsCNN
from constants import defaults

app = Flask(__name__)

classes = ["high", "low"]
cnn_storage = CNNSQL()

#cnn_storage.drop_table()
#cnn_storage = CNNSQL()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/prepare', methods=['POST'])
def prepare_for_cnn():
    
    img_size = defaults.IMAGE_SIZE
    prepared_data_save_path = defaults.PREPARED_TRAINING_DATA
    training_data_dir = defaults.TRAIN_DIR
    
    if('imageSize' in request.json):
        img_size = (int)(request.json['imageSize'])
    if('inputDirOption' in request.json):
        prepared_data_save_path = request.json['inputDirOption']
    if('preparedDataDirOption' in request.json):
        training_data_dir = request.json['preparedDataDirOption']
    
    Prepare.create_train_data(img_size, prepared_data_save_path, training_data_dir)
    
    return 'Prepared Successfully'

@app.route('/api/train', methods=['POST'])
def train_cnn():

    img_size = defaults.IMAGE_SIZE
    epochs = defaults.EPOCHS
    learning_rate = defaults.LEARNING_RATE
    model_save_path = defaults.MODEL_SAVE_PATH
    split = defaults.SPLIT
    model_attributes = {}
    
    if('split' in request.json):
        split = (float)(request.json['split'])

    if('imageSize' in request.json):
        img_size = (int)(request.json['imageSize'])
        
    if('epochs' in request.json):
        epochs = (int)(request.json['epochs'])
        
    if('learningRate' in request.json):
        learning_rate = (float)(request.json['learningRate'])
        
    if('modelSavePath' in request.json):
        model_save_path = request.json['modelSavePath']

    model_attributes['split']=split
    model_attributes['imageSize'] = img_size
    model_attributes['epochs'] = epochs
    model_attributes['learningRate'] = learning_rate
    model_attributes['modelSavePath'] = model_save_path
    model_attributes['name']="cnd-split{}-epochs{}-lr{}".format(split,epochs,learning_rate)
    
    CatsAndDogsNetwork = CatsAndDogsCNN(model_attributes)
    model_save_path = model_save_path + model_attributes['name']
    CatsAndDogsNetwork.train(model_save_path)
    
    cnn_storage.add_new_model(model_attributes)
    
    return 'Trained Successfully'

@app.route('/api/getModels', methods=['POST'])
def get_models():
    print("Current models are ",cnn_storage.get_names())
    return jsonify(cnn_storage.get_names())

@app.route('/api/test', methods=['POST'])
def test_cnn():

    fileName = request.json['fileName']
    expectedClass = request.json['expectedClass']
    modelName = request.json['modelName']
    clusters = (int)(request.json['clusters'])
    
    model_attributes = cnn_storage.get_attributes(modelName)
    
    CatsAndDogsNetwork = CatsAndDogsCNN(model_attributes)
    
    prediction_array = CatsAndDogsNetwork.test(modelName, fileName, model_attributes['imageSize'])
    
    prediction_prob = prediction_array[0]
    
    if prediction_prob[0] > prediction_prob[1]:
        return 'CAT'
    else:
        return 'DOG'
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008, debug=True)
