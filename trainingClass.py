# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:06:21 2019

@author: Vito
"""
from GMM import GMM

class trainingClass:
        
    def __init__(self, size, output_wavefile):
        self.Training_info = GMM(size, output_wavefile);
    
    def Training_feature_Mean(self):
    
        Mean_training = self.Training_info.GMM_Model_Mean()
    
        return Mean_training
    
    
    def Training_feature_Weight(self):
    
        Weight_training = self.Training_info.GMM_Model_Weight()
    
        return Weight_training
    
    
    def Training_feature_Covar(self):
    
        Covar_training = self.Training_info.GMM_Model_Covar()
    
        return Covar_training
    
    def adjustFeatures(self,name,mainF):
    
        self.Training_info.adjustFeatures(name,mainF);
        
