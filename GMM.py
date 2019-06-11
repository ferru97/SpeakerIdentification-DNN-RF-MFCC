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

    def __init__(self, M, input_audio_file):
        self.M = M
        self.Model = None
        self.nome_file = input_audio_file
        self.ImmFE = immFE.ImageFE();
        """
        if os.path.isfile(self.nome_file + '.bin'):
            self.Model = pickle.load(open(self.nome_file + '.bin', 'rb'))"""
        (rate, sig) = wav.read(input_audio_file)  # get audio data and sampling rate
        print("LINGHEZZA"+str(np.array(sig).shape[0])+" FREQ"+str(rate))
        
        if os.path.isfile(self.nome_file + '.npy'):
            feature_vectors = np.load(self.nome_file + '.npy');
        else:
            feature_vectors = mfcc.mfcc_features(sig)  # get feature marix of audio data
            feature_vectors = self.ImmFE.addImmFeatures(sig,rate,feature_vectors);
        
        save = np.array(feature_vectors);
        np.save(self.nome_file, save);
        
        print(np.array(feature_vectors).shape);
        self.feature = feature_vectors;

    def GMM_Model_Mean(self):
        if self.Model == None:
            self.Model = mixture.GaussianMixture(n_components=self.M, covariance_type="diag", n_init=10).fit(self.feature_training)
            pickle.dump(self.Model, open(self.nome_file + '.bin', 'wb'))
        mean = self.Model.means_

        return mean

    def GMM_Model_Weight(self):
        if self.Model == None:
            self.Model = mixture.GaussianMixture(n_components=self.M, covariance_type="diag", n_init=10).fit(self.feature_training)
            pickle.dump(self.Model, open(self.nome_file + '.bin', 'wb'))
        weight = self.Model.weights_

        return weight

    def GMM_Model_Covar(self):
        if self.Model == None:
            self.Model = mixture.GaussianMixture(n_components=self.M, covariance_type="diag", n_init=10).fit(self.feature_training)
            pickle.dump(self.Model, open(self.nome_file + '.bin', 'wb'))
        covar = self.Model.covariances_
        
        return covar
    
    def adjustFeatures(self,name,mainF):
        self.feature_training = self.feature[0:self.feature.shape[0],mainF];
        print("NUOVA FORMA:"+str(np.array(self.feature_test).shape));
        