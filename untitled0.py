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


def getLowFeatures(audio_save, estimators):
    
    size = 0;
    cols = 0;
    for i in range(0,audio_save.shape[0]):
        f = np.load(audio_save);
        size = size + f.shape[0];
        cols = f.shape[1] - 13; 
        
    dataset_x = np.zeros((size,cols));
    dataset_y = np.zeros((size,1));         
    
    index = 0;
    for i in range(0,audio_save.shape[0]):
        f = np.load(audio_save);
        f = f[0:f.shape[0],13:f.shape[1]];
        y = np.ones((f.shape[0],1))*i;
        
        before = index + f.shape[0];
        index = before - 1;
    
        dataset_x = np.insert(dataset_x, index, f, 0);
        dataset_y = np.insert(dataset_y, index, y, 0);
    
    X_train,y_train,X_test,y_test = train_test_split(dataset_x,dataset_y,test_size=0.3, random_state=2)
    
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 1))
    sel.fit(X_train, X_test)
    
    sel.get_support()
    
    dt = pd.DataFrame(X_train);
    selected_feat = np.array(dt.columns[(sel.get_support())])
    
    len(selected_feat)
    print(selected_feat)
    
    n_feat = len(selected_feat);
    X_train_new = X_train[0:X_train.shape[0],selected_feat];
    return X_train_new;
    
    


    
    
    


