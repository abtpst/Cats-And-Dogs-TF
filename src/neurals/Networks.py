'''
Created on Nov 27, 2017

@author: abhijit.tomar
'''
import tensorflow as tf
import tflearn
import os
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from constants import defaults
from data_process import Prepare
class CatsAndDogsCNN(object):
    '''
    classdocs
    '''
    
    def __init__(self, params=None):
        
        self.img_size = params['imageSize']
        self.learning = params['learningRate']
        if 'modelSavePath' in params:
            self.save_location = params['modelSavePath']
        else:
            self.save_location = defaults.MODEL_SAVE_PATH
        self.epochs = params['epochs']
        
        tf.reset_default_graph()
             
        convnet = input_data(shape=[None, self.img_size, self.img_size, 1], name='input')
        
        convnet = conv_2d(convnet, 32, 2, activation='relu')       
        convnet = max_pool_2d(convnet, 2)
                
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
                
        convnet = conv_2d(convnet, 32, 2, activation='relu')            
        convnet = max_pool_2d(convnet, 2)
                
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
                
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
                
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
                
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
                
        convnet = fully_connected(convnet, 2, activation='softmax')
        self.convnet = regression(convnet, optimizer='adam', learning_rate=self.learning, loss='categorical_crossentropy', name='targets')
                
        self.model = tflearn.DNN(self.convnet, tensorboard_dir=defaults.TF_LOGS)
            
    def train(self, model_save_path):
        
        if os.path.exists(model_save_path + ".meta"):
            self.model.load(model_save_path)
            print('model loaded')
        
        X, Y, test_x, test_y = Prepare.get_data_for_fitting(img_size=self.img_size)
        model_name = model_save_path[model_save_path.rindex('/') + 1:]
        self.model.fit({'input': X}, {'targets': Y}, n_epoch=self.epochs, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=model_name)
    
        self.model.save(model_save_path)
    
    def test(self, model_save_path, img_name, img_size):
        
        if os.path.exists(model_save_path + ".meta"):
            self.model.load(model_save_path)
            print('model loaded')
        
        img_path = defaults.TEST_IMAGES + img_name
        
        test_input = Prepare.format_single_image_for_testing(img_path, img_size)
        
        pred = self.model.predict(test_input)
        
        return pred
