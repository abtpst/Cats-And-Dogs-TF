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
    
def create_train_data(IMG_SIZE):
    
    train_data = []
    
    for img in tqdm(os.listdir(defaults.TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(defaults.TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img),np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy',train_data)
    return train_data

def process_test_data(IMG_SIZE):
    test_data = []
    
    for img in tqdm(os.listdir(defaults.TEST_DIR)):
        path = os.path.join(defaults.TEST_DIR,img)
        
        img_num = img.split('.')[0]
        
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        
        test_data.append([np.array(img),img_num])
        
        np.save('test_data.npy',test_data)
        return test_data
    
def get_data_for_fitting(IMG_SIZE):
    
    train_data = np.load('train_data.npy')
    
    train = train_data[:-500]
    test = train_data[-500:]
        
    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]
        
    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]
    
    return X,Y,test_x,test_y 