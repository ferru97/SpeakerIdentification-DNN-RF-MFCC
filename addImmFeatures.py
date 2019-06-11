# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:02:54 2019

@author: Vito
"""

import numpy as np
import vgg19FE as vggfe
import saveSpecImage as specImm


class ImageFE:
    fe = None;

    def __init__(self):
       self.fe = vggfe.vgg16FE(extraction_layer="block5_pool");
       self.fe.load_model();

    def addImmFeatures(self,signal,fs,matrix, winlenMS=20, winstepMS=10):
        
        start = 0;
        step = int((fs/1000)*winlenMS);
        overlap = int((fs/1000)*winstepMS);
        new_features = np.zeros((matrix.shape[0],25088));
        row = 0;
        end = signal.shape[0];
        while start < end:
    
            stop = start + step; 
            self.printProgressBar(stop, end, prefix = 'Progress:', suffix = 'Complete', length = 50)
            if stop > signal.shape[0]-1:
                break;
            
            sub = signal[start:stop];
            image = specImm.save_spec_image(sub,fs);
            new_features[row,:] = self.fe.extract_features(image);
            
            start = start + overlap ;
            row = row + 1;
    

        output = np.concatenate((matrix,new_features), axis=1);
        output = output[0:row, 0:output.shape[1]]
        return output;
    
    # Print iterations progress
    def printProgressBar (self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()
        
        
        
    