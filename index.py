import os
"""
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
"""

import struct
import sys
import time
import wave
from threading import Thread
from warnings import filterwarnings

import numpy as np
import pyaudio
import scipy.io.wavfile as wav


import Analysis
import mfcc as mfcc_old
import training

import trainingClass
from joblib import Parallel, delayed
import addImmFeatures as immFE
from multiprocessing import Process
import matplotlib.pyplot as plt

import scrapeFeatures as scrape

import GMM as gmm

filterwarnings('ignore')
info = dict()


class MyApp():

    # noinspection PyArgumentList
    def MyApp(self):
        # inizializzazioni
        self.n = 0
        self.names = list()
        self.p_weight = list()
        self.mean = list()
        self.covar = list()
		

    def load_samples_event(self):
        self.names = ['Dataset voci/Confessioni di una mente pericolosa_15sec/MusicaConfession_15sec', 
                      'Dataset voci/Confessioni di una mente pericolosa_15sec/RockwellConfession_15sec', 
                      'Dataset voci/Confessioni di una mente pericolosa_15sec/SilenzioConfessions_15sec', 
                      'Dataset voci/Confessioni di una mente pericolosa_15sec/SergenteConfesion_15sec']
        self.n = len(self.names)

        self.names_test = ['Audio/ConfessionsOfdangerous_test10low']

    # noinspection PyUnusedLocal
    def train(self):
        self.p_weight = [0 for i in range(len(self.names))]
        self.mean = [0 for i in range(len(self.names))]
        self.covar = [0 for i in range(len(self.names))]


        objects = [];
        print("Estrazione Fetures Originali")
        for name in self.names:
            print("File: "+name)
            trainer = trainingClass.trainingClass(32,name + '.wav')
            objects.append(trainer);
           
        print("OK")
        if os.path.isfile(self.nome_file + 'good_feat_index.npy'):    
            selected_feat = np.load(self.nome_file + 'good_feat_index.npy');    
        else:
            selected_feat = scrape.scrapeFeatures(self.names,10,2);
            np.save(self.nome_file+'good_feat_index', np.array(selected_feat));
            
            print("Ottimizzazione Fetures")
            idx = 0
            for name in self.names:
                print("File: "+name)
                trainer2 = objects.pop();
                trainer2.adjustFeatures(name,selected_feat)
                self.p_weight[idx] = trainer2.Training_feature_Weight()
                self.mean[idx] = trainer2.Training_feature_Mean()
                self.covar[idx] = trainer2.Training_feature_Covar()
                idx += 1
           
            
    def exstract_test_features(self):
        self.p_weight_test = [0 for i in range(len(self.names_test))]
        self.mean_test = [0 for i in range(len(self.names_test))]
        self.covar_test = [0 for i in range(len(self.names_test))]


        idx = 0
        objects = [];
        print("Estrazione Fetures Originali")
        for name in self.names_test:
            print("File: "+name)
            trainer = trainingClass.trainingClass(32,name + '.wav')
            objects.append(trainer);
            idx += 1
            """
        if os.path.isfile(self.nome_file + 'good_feat_index.npy'):    
            selected_feat = np.load(self.nome_file + 'good_feat_index.npy');    
        else:
            selected_feat = scrape.scrapeFeatures(self.names,10,2);
            np.save(self.nome_file+'good_feat_index', np.array(selected_feat));
            
            print("Ottimizzazione Fetures")
            idx = 0
            for name in self.names:
                print("File: "+name)
                trainer2 = objects.pop();
                trainer2.adjustFeatures(name,selected_feat)
                self.p_weight[idx] = trainer2.Training_feature_Weight()
                self.mean[idx] = trainer2.Training_feature_Mean()
                self.covar[idx] = trainer2.Training_feature_Covar()
                idx += 1
            """

        

    def test_event(self, blocksize=20000, width=2, channels=1, rate=44100):
        
        
                
        """
        good = 0;
        bad = 0;
        count = 0;
        
        for testFeature in self.testFeatures: 
            print("TEST FILE "+str(self.names[count]))
            end = 0;
            stop = False;
            while end<testFeature.shape[0] and stop==False:
               start = end;
               end = end + 30;
               print("RIGHE "+str(testFeature.shape[0]) + " END "+str(end)) 
               if(end-1 > testFeature.shape[0]):
                   stop = True;
               else: 
                   sub = testFeature[start:end-1,0:testFeature.shape[1]]
                   
                   predict = Analysis.GMM_identity(sub,len(self.names),self.names,self.p_weight,self.mean,self.covar)   
                 
                   if predict == count:
                       good = good + 1;
                   else:
                       bad = bad + 1;
        count = count + 1;           
               
        print("\nGOOD: "+str(good));
        print("\nBAD: "+str(bad));
                
        
        self.btnTest.setEnabled(False)
        
        (rate, sig) = wav.read(self.names[0]+".wav") 
        (rate2, sig2) = wav.read(self.names[1]+".wav") 
        y1 = sig[0:blocksize];
        y2 = sig2[0:blocksize];
        
        ImmFE = immFE.ImageFE();
        
        x1 = mfcc_old.mfcc_features(np.array(y1))
        x1 = ImmFE.addImmFeatures(y1,rate,x1)
        print(np.array(x1).shape)
        plt.ylabel('some numbers')
        plt.show()
        x2 = mfcc_old.mfcc_features(np.array(y2))
        x2 = ImmFE.addImmFeatures(y2,rate,x2)
        print(np.array(x2).shape)
    
        final1 = Analysis.GMM_identity(x1,len(self.names),self.names,self.p_weight,self.mean,self.covar)

        "final2 = Analysis.GMM_identity(x2,len(self.names),self.names,self.p_weight,self.mean,self.covar)"
        print("PREDIZIONE: "+ self.names[final1]+"/"+self.names[0]);
          print("PREDIZIONE: "+ self.names[final2]+"/"+self.names[1]);
        """
    
go = MyApp();
go.load_samples_event();
go.train();
