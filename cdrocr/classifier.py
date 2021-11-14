#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import re
#-------------------------
# imports
#-------------------------
import tensorflow as tf
import os
import numpy as np
import cv2 
from .utils import *
class Language(object):
    def __init__(self,
                lang,
                weights="models/crnn/lang.h5",
                img_height=64,
                img_width=512,
                nb_channels=1,
                device="cpu"):

        
        self.labels     =   ["bangla","english"]
        self.img_height =   img_height
        self.img_width  =   img_width
        self.nb_channels=   nb_channels
        self.device     =   device
        self.weights    =   weights
        self.model=tf.keras.applications.DenseNet121(input_shape=(64,512,1),
                                        classes=2,
                                        pooling="avg",
                                        weights=None)
        if self.device=="cpu":
            self.strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
            with self.strategy.scope():
                self.model.load_weights(self.weights)
        else:
            self.model.load_weights(self.weights)
        LOG_INFO("Loaded Language classifier")
    
    def infer(self,img):
        '''
            args:
                img =  3 channel non-paded image
        '''
        img,_=padWords(img,(self.img_height,self.img_width),ptype="left")
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h,w=img.shape
        assert h==self.img_height
        assert w==self.img_width
        img=np.expand_dims(img,axis=-1)
        img=np.expand_dims(img,axis=0)
        img=img/255
        pred=self.model.predict(img)[0]
        label=self.labels[np.argmax(pred)]
        return img,label