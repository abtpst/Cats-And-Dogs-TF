'''
Created on Nov 27, 2017

@author: abhijit.tomar
'''
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from constants import defaults

def label_img(img):
    
    word_label = img.split('.')[-3]
    
    if word_label == 'cat' : 
        return [1,0]
    elif word_label == 'dog' : 
        return [0,1]
    
def create_train_data(img_size=defaults.IMAGE_SIZE,prepared_data_save_path=defaults.PREPARED_TRAINING_DATA,training_data_dir=defaults.TRAIN_DIR):
    
    train_data = []
    
    for img in tqdm(os.listdir(training_data_dir)):
        label = label_img(img)
        path = os.path.join(training_data_dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
        train_data.append([np.array(img),np.array(label)])
    shuffle(train_data)
    np.save(prepared_data_save_path,train_data)
    return train_data

def process_test_data(img_size=defaults.IMAGE_SIZE,test_dir=defaults.TEST_DIR,test_data_save_path=defaults.PREPARED_TEST_DATA):
    test_data = []
    
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir,img)
        
        img_num = img.split('.')[0]
        
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
        
        test_data.append([np.array(img),img_num])
        
        np.save('test_data_save_path',test_data)
        return test_data
    
def get_data_for_fitting(img_size=defaults.IMAGE_SIZE,prepared_data_save_path=defaults.PREPARED_TRAINING_DATA):
    
    train_data = np.load(prepared_data_save_path)
    
    train = train_data[:-500]
    test = train_data[-500:]
        
    X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
    Y = [i[1] for i in train]
        
    test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
    test_y = [i[1] for i in test]
    
    return X,Y,test_x,test_y 