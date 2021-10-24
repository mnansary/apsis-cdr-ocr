#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import tensorflow as tf
import os
import numpy as np
import cv2 
from .utils import *
import matplotlib.pyplot as plt
#-------------------------
# model
#------------------------
class Classifier(object):
    def __init__(self,
                model_weights,
                img_dim=(256,256,3),
                num_classes=2,
                labels=["nid","smart"]):
        # Classifier Initially trained with unet
        self.img_dim=img_dim
        self.num_classes=num_classes
        self.labels=labels
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        with strategy.scope():
            self.model=tf.keras.applications.DenseNet121(input_shape=self.img_dim,classes=num_classes,weights=None)
            # load weights        
            self.model.load_weights(model_weights)
    
    def process(self,img):
        # process
        img=cv2.resize(img,(self.img_dim[1],self.img_dim[0]))
        img=img/255.0
        data=np.expand_dims(img,axis=0)
        # predict
        pred=self.model.predict(data)
        card_type=self.labels[np.argmax(pred[0])]
        return card_type

