#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import re
#----------------
# imports
#---------------
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers
import os 
from .utils import *
from itertools import groupby
# globals
en_vocab=["","0","1","2","3","4","5","6","7","8","9"]
bn_vocab=["",'০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']
#----------------------------------------------------------------------------------------
# model defs
#----------------------------------------------------------------------------------------
class CRNN(object):
    def __init__(self,
                lang,
                wdir="models/crnn/",
                img_height=64,
                img_width=512,
                nb_channels=1,
                pos_max=15,
                level=4,
                lstms=128,
                device="cpu"):

        
        if lang=="bangla":
            self.vocab      =   bn_vocab
            self.weights    =   os.path.join(wdir,"bn_num.h5")
        
        else:
            self.vocab      =   en_vocab
            self.weights    =   os.path.join(wdir,"en_num.h5")

        self.img_height =   img_height
        self.img_width  =   img_width
        self.nb_channels=   nb_channels
        self.pos_max    =   pos_max
        self.level      =   level
        self.lstms      =   lstms
        self.device     =   device
            
        self.build_model()
        LOG_INFO(f"Loaded Weights:{lang}")

    def build_model(self):
        inp=tf.keras.layers.Input(shape=(self.img_height,self.img_width,self.nb_channels))
        x = tf.keras.layers.Permute((2, 1, 3))(inp)
        feat=tf.keras.applications.DenseNet121(input_tensor=x,weights=None,include_top=False)
        x=feat.get_layer(name=f"pool{self.level}_conv").output
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        # reshape
        bs,d1,d2,d3=x.shape
        reshape_dim=(d1,int(d2*d3))
        x = tf.keras.layers.Reshape(reshape_dim)(x) 
        
        # bi-lstm
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.lstms, return_sequences=True), name='bi_lstm1')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.lstms, return_sequences=True), name='bi_lstm2')(x)
        # logits
        logits = layers.Dense(units=len(self.vocab), name='logits')(x)
        self.model= tf.keras.Model(inputs=inp, outputs=logits)
        if self.device=="cpu":
            self.strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
            with self.strategy.scope():
                self.model.load_weights(self.weights)
        else:
            self.model.load_weights(self.weights)
    
    def ctc_decoder(self,pred):
        '''
        input: given batch of predictions from text rec model
        output: return lists of raw extracted text

        '''
        text_list = []
        pred_vocab= []
        pred_indcies = np.argmax(pred, axis=2)
        
        for i in range(pred_indcies.shape[0]):
            ans = ""
            _vocab=[]
            ## merge repeats
            merged_list = [k for k,_ in groupby(pred_indcies[i])]
            
            ## remove blanks
            for p in merged_list:
                if p != 0:
                    ans += self.vocab[int(p)]
                    _vocab.append(self.vocab[int(p)])
        
            text_list.append(ans)
            pred_vocab.append(_vocab)
            
        return text_list
    
    def infer(self,img):
        '''
            args:
                img:   cropped paded image 
        '''
        # 3 channel pad
        pred=self.model.predict(img)
        text=self.ctc_decoder(pred)[0]
        return text
        