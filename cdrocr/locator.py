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
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
#-------------------------
# model
#------------------------

class Locator(object):
    def __init__(self,weights_path,img_dim=(256,256,1),bbox_class=8):
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        self.img_dim=img_dim
        self.bbox_class=bbox_class
        # loading weights
        weights_path=os.path.join(weights_path,"loc")
        seg_weights =os.path.join(weights_path,"segment.h5")
        cor_weights =os.path.join(weights_path,"corner.h5")
        #  models
        with strategy.scope():
            self.seg=sm.Unet("densenet121",input_shape=self.img_dim, classes=1,encoder_weights=None)
            self.seg.load_weights(seg_weights)

            self.cor=self.cor_model()
            self.cor.load_weights(cor_weights)

    def cor_model(self):
        mod=tf.keras.applications.DenseNet121(input_shape=self.img_dim,weights=None,include_top=False)
        inp=mod.input
        x=mod.output
        x=tf.keras.layers.GlobalAveragePooling2D()(x)
        x=tf.keras.layers.Dense(self.bbox_class,activation=None)(x)
        model=tf.keras.Model(inputs=inp,outputs=x)
        return model

    def process(self,img):
        org=np.copy(img)
        h,w,d=org.shape
        
        img=remove_shadows(img)
        img=threshold_image(img,True)
        # process
        img=cv2.resize(img,(self.img_dim[1],self.img_dim[0]))
        img=np.expand_dims(img,axis=-1)
        img=img/255.0
        # tensor
        img=np.expand_dims(img,axis=0)
        # predict
        seg=self.seg.predict(img)
        
        pts=self.cor.predict(seg)[0]
        # coords
        coords=[]
        for i in range(0,8,2):
            x,y=pts[i:i+2]
            coords.append([int(x),int(y)])
        rx=w/self.img_dim[1]
        ry=h/self.img_dim[0]
        # convert source
        src=[]

        for c in coords:
            x,y=c
            x,y=int(x*rx),int(y*ry)
            src.append([x,y])
        
        
        src=np.float32(src)
        w=int(max(np.linalg.norm(src[0]-src[1]),np.linalg.norm(src[3]-src[2])))
        h=int(max(np.linalg.norm(src[0]-src[3]),np.linalg.norm(src[1]-src[2])))
        dst=np.float32([[0,0],[w,0],[w,h],[0,h]])
        M   = cv2.getPerspectiveTransform(src,dst)
        img = cv2.warpPerspective(org, M, (w,h))
        return img

