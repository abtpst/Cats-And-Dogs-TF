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

# cnn_storage.drop_table()
# cnn_storage = CNNSQL()
ConvNet = None
current_model_name = ""

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
    model_attributes = {}
    
    if('imageSize' in request.json):
        img_size = (int)(request.json['imageSize'])
        
    if('epochs' in request.json):
        epochs = (int)(request.json['epochs'])
        
    if('learningRate' in request.json):
        learning_rate = (float)(request.json['learningRate'])
        
    if('modelSavePath' in request.json):
        model_save_path = request.json['modelSavePath']
        model_save_path = model_save_path + "-" + str(learning_rate) + "-" + str(epochs)

    model_attributes['imageSize'] = img_size
    model_attributes['epochs'] = epochs
    model_attributes['learning'] = learning_rate
    model_attributes['modelSavePath'] = model_save_path
    
    CatsAndDogsNetwork = CatsAndDogsCNN(model_attributes)
    
    CatsAndDogsNetwork.train(model_save_path)
    
    return 'Trained Successfully'

@app.route('/api/getModels', methods=['POST'])
def get_models():

    return jsonify(cnn_storage.get_names())

@app.route('/api/test', methods=['POST'])
def test_cnn():

    fileName = request.json['fileName']
    expectedClass = request.json['expectedClass']
    modelName = request.json['modelName']
    clusters = (int)(request.json['clusters'])
    '''
    global ConvNet
    global current_model_name
    if ConvNet is None or current_model_name != modelName:
        model_attributes = cnn_storage.get_attributes(modelName)
        ConvNet = conv2(
                windowSize=model_attributes['windowSize'],
                stride=model_attributes['windowStride'],
                n_classes= len(classes),
                imageSize=model_attributes['imageSize'],
                runs=model_attributes['numRuns'],
                saveLocation=common.MODEL_LOCATION,
                keep=0.5,
                split=model_attributes['split'],
                epochs=model_attributes['epochs'],
                learningRate=model_attributes['learningRate'])
        current_model_name = modelName

   
    analysis = ConvNet.analyze_image("../resources/test/"+expectedClass+"/"+fileName,modelName,clusters)
    
    return jsonify(analysis)
    '''
    return ''
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008, debug=True)
