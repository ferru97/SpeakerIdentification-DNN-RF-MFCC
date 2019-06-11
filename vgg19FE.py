from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa



class vgg16FE:
    
    extraction_layer = None;
    imageH = None;
    imageW = None;
    model = None;
        
    def __init__(self, extraction_layer, imageH=224, imageW=224):
        self.extraction_layer = extraction_layer;
        self.imageH = imageH;
        self.imageW = imageW;
    
    def load_model(self,weights='imagenet'):
        base_model = VGG19(weights)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.extraction_layer).output);
    
    def extract_features(self,imageEx):
        x = image.img_to_array(imageEx);
        x = np.expand_dims(x, axis=0);
        x = preprocess_input(x);
            
        return np.array(self.model.predict(x)).ravel();
          