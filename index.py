import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import struct
import sys
import time
import wave
from threading import Thread
from warnings import filterwarnings

import numpy as np
import pyaudio
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import *
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


class MyApp(QMainWindow):
    msg_trigger = pyqtSignal(str)
    test_trigger = pyqtSignal(str, int)

    # noinspection PyArgumentList
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi('srgui.ui', self)
        """self.show()"""
        # triggers
        self.msg_trigger.connect(self.handle_msg_trigger)
        self.test_trigger.connect(self.handle_test_trigger)
        info['msg_trigger'] = self.msg_trigger
        # inizializzazioni
        self.n = 0
        self.names = list()
        self.p_weight = list()
        self.mean = list()
        self.covar = list()
        # bind eventi
        self.btnSamples.clicked.connect(lambda: self.load_samples_event())
        self.btnTest.clicked.connect(lambda: self.test_event())
        self.btnNewSession.clicked.connect(lambda: self.new_session_event())
        self.btnRecAudio.clicked.connect(lambda: self.rec_audio_event())
        self.btnFileAudio.clicked.connect(lambda: self.load_wav_files())
        self.btnStartTrain.clicked.connect(lambda: self.train())
        
        self.load_samples_event()
        self.train();

    def handle_msg_trigger(self, text):
        self.textArea.appendPlainText(text)

    def handle_test_trigger(self, text, num):
        self.textArea.appendPlainText(text)
        self.lblTime.setText(self.lblTime.text()[:19] + str(num))

    def switch_train_functions(self, flag):
        self.btnSamples.setEnabled(flag)
        self.btnFileAudio.setEnabled(flag)
        self.btnRecAudio.setEnabled(flag)

    def new_session_event(self):
        self.switch_train_functions(True)
        self.btnTest.setEnabled(False)
        self.names.clear()
        self.textArea.clear()
        self.textAreaNomi.clear()
        self.textAreaNomi.setEnabled(False)
        self.btnStartTrain.setEnabled(False)

    def load_samples_event(self):
        self.names = ['Dataset Voci/Confessioni di una mente pericolosa_15sec/MusicaConfession_15sec', 
                      'Dataset Voci/Confessioni di una mente pericolosa_15sec/RockwellConfession_15sec', 
                      'Dataset Voci/Confessioni di una mente pericolosa_15sec/SergenteConfesion_15sec', 
                      'Dataset Voci/Confessioni di una mente pericolosa_15sec/SilenzioConfessions_15sec']
        self.n = len(self.names)
        self.train()
        self.switch_train_functions(False)
        
        self.names_test = ['Voci/ConfessionsOfdangerous_test10min']

    # noinspection PyArgumentList
    def load_wav_files(self):
        wav_filter = "WAV (*.wav)"
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        files = file_dialog.getOpenFileNames(self, "Open files", filter=wav_filter)
        for name in files[0]:
            if name != str():
                self.textAreaNomi.appendPlainText('- ' + name)
                self.names.append(name[:-4])
            self.textAreaNomi.setEnabled(True)
            if len(self.names) > 1:
                self.btnStartTrain.setEnabled(True)

    def rec_audio_event(self, record_seconds=10):
        dialog = InsertNameDialog()
        name = dialog.result
        self.textAreaNomi.setEnabled(True)
        if name != str():
            self.textAreaNomi.appendPlainText('- ' + name)
            self.names.append(name)

            if len(self.names) > 1:
                self.btnStartTrain.setEnabled(True)

            self.textArea.appendPlainText(
                'Registrazione audio per {0:.3f} secondi...'.format(record_seconds))

            def record(blocksize=64, width=2, channels=1, rate=44100):
                global info

                p1 = pyaudio.PyAudio()

                stream1 = p1.open(format=p1.get_format_from_width(width),
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  output=False)

                output_wavefile = (name + '.wav')
                output_wf = wave.open(output_wavefile, 'w')
                output_wf.setframerate(rate)
                output_wf.setsampwidth(width)
                output_wf.setnchannels(channels)

                num_blocks = int(rate / blocksize * record_seconds)

                # Start loop
                for i in range(0, num_blocks):
                    input_string = stream1.read(blocksize)

                    output_wf.writeframes(input_string)

                stream1.stop_stream()
                stream1.close()
                p1.terminate()
                info['msg_trigger'].emit('Registrazione completata!')

            thread = Thread(target=record)
            thread.start()

    # noinspection PyUnusedLocal
    def train(self):
        self.p_weight = [0 for i in range(len(self.names))]
        self.mean = [0 for i in range(len(self.names))]
        self.covar = [0 for i in range(len(self.names))]


        idx = 0
        objects = [];
        print("Estrazione Fetures Originali")
        for name in self.names:
            print("File: "+name)
            trainer = trainingClass.trainingClass(32,name + '.wav')
            objects.append(trainer);
            idx += 1
            
        selected_feat = scrape.scrapeFeatures(self.names,10,2);
            
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
            
        print("Features Test")
        gmm.GMM(32,self.names_test);

        # noinspection PyCallByClass,PyArgumentList
        QMessageBox.information(
            QWidget(),
            "Training",
            "Modello predittivo creato!")
        self.textArea.appendPlainText('Training: modello predittivo creato!')
        self.textArea.appendPlainText('Ora puoi effettuare il test :)\n')
        self.btnTest.setEnabled(True)
        self.btnTest.show()
        

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
     

class InsertNameDialog(QDialog):
    # noinspection PyArgumentList
    def __init__(self, parent=None):
        super(InsertNameDialog, self).__init__(parent)
        uic.loadUi('name_dialog.ui', self)
        self.result = str()
        self.init_gui()

    def init_gui(self):
        self.btnConferma.clicked.connect(lambda: self.continue_event())
        self.btnAnnulla.clicked.connect(lambda: self.close())
        self.exec_()

    def continue_event(self):
        self.result = self.txtNome.text()
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())
