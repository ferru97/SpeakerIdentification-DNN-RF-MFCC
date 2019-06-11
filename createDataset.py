# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:03:54 2019

@author: Vito
"""
import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


def createDataset(audio_save, rand_state):
    
    size = 0;
    cols = 0;
    for name in audio_save:
        f = np.load(str(name)+".wav.npy");
        size = size + f.shape[0];
        cols = f.shape[1] - 13; 
        
    dataset_x = np.zeros((size,cols));
    dataset_y = np.zeros((size,1));         
    
    index = 0;
    i = 0;
    for name in audio_save:
        f = np.load(str(name)+".wav.npy");
        f = f[0:f.shape[0],13:f.shape[1]];
        y = np.ones((f.shape[0],1))*i;
        i = i + 1;
        
        before = index + f.shape[0];
        index = before - 1;
    
        dataset_x = np.insert(dataset_x, index, f, 0);
        dataset_y = np.insert(dataset_y, index, y, 0);
    
    X_train,y_train,X_test,y_test = train_test_split(dataset_x,dataset_y,test_size=0.01, random_state=rand_state)
    
    return (X_train,y_train,X_test,y_test);