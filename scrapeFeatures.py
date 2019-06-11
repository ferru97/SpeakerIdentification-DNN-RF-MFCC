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
import createDataset as cdata


def scrapeFeatures(audio_save, estimators,r_state):
    
    X_train,y_train,X_test,y_test = cdata.createDataset(audio_save,r_state)
    
    sel = SelectFromModel(RandomForestClassifier(n_estimators = estimators))
    sel.fit(X_train, X_test)
    
    sel.get_support()
    
    dt = pd.DataFrame(X_train);
    selected_feat = np.array(dt.columns[(sel.get_support())])
    
    len(selected_feat)
    print(selected_feat)
    
    return selected_feat;
