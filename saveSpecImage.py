# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:31:32 2019

@author: Vito
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from PIL import Image

def save_spec_image(samples,sample_rate):
    name = "temp_spec.jpg";
    NFFT = 26  # the length of the windowing segments
    Fs = sample_rate  # the sampling frequency
    Window = signal.hamming(NFFT);
    Noverlap = round(NFFT/2);
        
    frequencies, times, spectrogram = signal.spectrogram(samples, fs=Fs, window=Window, noverlap=Noverlap ,nfft=NFFT)
    
    plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram))
    plt.axis('off');
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.savefig(name);
    
    im1 = Image.open(name);
    # adjust width and height to your needs
    width = 224;
    height = 224;
    im2 = im1.resize((width, height), Image.NEAREST)    
    
    return im2;

