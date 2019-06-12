""" This file contains a GMM class, which also calculates the means, weights and covariances of each
Gaussian model. In addition, the file return those values for further study.

Author: Alessandro Gerardi
"""

import numpy as np
from sklearn import mixture
import scipy.io.wavfile as wav
import mfcc
import pickle
import os
import addImmFeatures as immFE

class GMM:
    ImmFE = None
    

    def __init__(self, M, input_audio_file):
        self.M = M
        self.Model = None
        self.nome_file = input_audio_file
     
        if os.path.isfile(self.nome_file + '_model.bin'):
            self.Model = pickle.load(open(self.nome_file + '.bin', 'rb'))
        else:
            (rate, sig) = wav.read(input_audio_file)  # get audio data and sampling rate
            
            if os.path.isfile(self.nome_file + 'good_feat.npy'):
                self.feature_training = np.load(self.nome_file + 'good_feat_index.npy');
                print("Good Features Loaded");
            else:
                if os.path.isfile(self.nome_file + '.npy'):
                    feature_vectors = np.load(self.nome_file + '.npy');
                else:
                    print("Estrazione Features");
                    feature_vectors = mfcc.mfcc_features(sig)  # get feature marix of audio data
                    if(self.ImmFE == None):
                        self.ImmFE = immFE.ImageFE();
                    feature_vectors =  self.ImmFE.addImmFeatures(sig,rate,feature_vectors);
                    np.save(self.nome_file, np.array(feature_vectors));
                    print(np.array(feature_vectors).shape);
                    self.feature = feature_vectors;

    def GMM_Model_Mean(self):
        if self.Model == None:
            self.Model = mixture.GaussianMixture(n_components=self.M, covariance_type="diag", n_init=10).fit(self.feature_training)
            pickle.dump(self.Model, open(self.nome_file + '_model.bin', 'wb'))
        mean = self.Model.means_

        return mean

    def GMM_Model_Weight(self):
        if self.Model == None:
            self.Model = mixture.GaussianMixture(n_components=self.M, covariance_type="diag", n_init=10).fit(self.feature_training)
            pickle.dump(self.Model, open(self.nome_file + '_model.bin', 'wb'))
        weight = self.Model.weights_

        return weight

    def GMM_Model_Covar(self):
        if self.Model == None:
            self.Model = mixture.GaussianMixture(n_components=self.M, covariance_type="diag", n_init=10).fit(self.feature_training)
            pickle.dump(self.Model, open(self.nome_file + '_model.bin', 'wb'))
        covar = self.Model.covariances_
        
        return covar
    
    def adjustFeatures(self,name,mainF):
        if self.Model != None:
            self.feature_training = self.feature[0:self.feature.shape[0],mainF];
        print("NUOVA FORMA:"+str(np.array(self.feature_test).shape));
        