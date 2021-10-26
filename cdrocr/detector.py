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
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
from .utils import LOG_INFO,padDetectionImage
import matplotlib.pyplot as plt
#-------------------------
# model
#------------------------
class BaseDetector(object):
    def __init__(self,
                model_weights,
                img_dim,
                data_channel,
                backbone='densenet121',
                use_cpu=True):
        # craft with unet
        self.img_dim=img_dim
        self.data_channel=data_channel
        self.backbone=backbone
        if use_cpu:
            strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
            with strategy.scope():
                self.model=sm.Unet(self.backbone,input_shape=self.img_dim, classes=self.data_channel,encoder_weights=None)
                self.model.load_weights(model_weights)
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.model=sm.Unet(self.backbone,input_shape=self.img_dim, classes=self.data_channel,encoder_weights=None)
            self.model.load_weights(model_weights)


class CRAFT(BaseDetector):
    def __init__(self, model_weights, img_dim=(1024,1024,3), data_channel=2, backbone="densenet121",use_cpu=True):
        super().__init__(model_weights, img_dim, data_channel, backbone=backbone,use_cpu=use_cpu)
        LOG_INFO("Loaded Detection Model,craft",mcolor="green")
    
    def detect(self,img,det_thresh=0.4,text_thresh=0.7,debug=False):
        '''
        detects words from an image
        args:
            img  : rgb image
        '''
        # squre img
        img,cfg=padDetectionImage(img)
        # for later
        org_h,org_w,d=img.shape
        # predict
        data=cv2.resize(img,(self.img_dim[0],self.img_dim[1]))
        # predict
        data=np.expand_dims(data,axis=0)
        data=data/255
        pred=self.model.predict(data)[0]
        # decode
        linkmap=np.squeeze(pred[:,:,1])
        textmap=np.squeeze(pred[:,:,0])
        img_h,img_w=textmap.shape

        _, text_score = cv2.threshold(textmap,
                                    thresh=det_thresh,
                                    maxval=1,
                                    type=cv2.THRESH_BINARY)
        _, link_score = cv2.threshold(linkmap,
                                    thresh=det_thresh,
                                    maxval=1,
                                    type=cv2.THRESH_BINARY)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(text_score + link_score, 0, 1).astype('uint8'),connectivity=4)
        if debug:
            plt.imshow(labels)
            plt.show()
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < 10:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < text_thresh:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key] for key in
                [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))
            # idx 
            segmap=cv2.resize(segmap,(org_w,org_h))
            idx = np.where(segmap>0)            
            y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0])+1, np.min(idx[1]), np.max(idx[1])+1
            boxes.append([x_min,y_min,x_max,y_max])
        return boxes

class Locator(BaseDetector):
    def __init__(self, model_weights, img_dim=(1024,1024,3), data_channel=2, backbone="densenet121",use_cpu=True):
        super().__init__(model_weights, img_dim, data_channel, backbone=backbone,use_cpu=use_cpu)
        LOG_INFO("Loaded Detection Model,Locator",mcolor="green")
    
    def detect(self,img):
        '''
        detects regions from an image
        args:
            img  : rgb image
        '''
        # squre img
        img,cfg=padDetectionImage(img)
        # for later
        org_h,org_w,d=img.shape
        # predict
        data=cv2.resize(img,(self.img_dim[0],self.img_dim[1]))
        # predict
        data=np.expand_dims(data,axis=0)
        data=data/255
        pred=self.model.predict(data)[0]
        return pred