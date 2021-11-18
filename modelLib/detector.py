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
from .utils import *
import matplotlib.pyplot as plt
from scipy.special import softmax
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
        ref_boxes=[]
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
            

            # make box
            np_temp = np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            np_contours = np_temp.transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # boundary check due to minAreaRect may have out of range values 
            # (see https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9)
            for p in box:
                if p[0] < 0:
                    p[0] = 0
                if p[1] < 0:
                    p[1] = 0
                if p[0] >= img_w:
                    p[0] = img_w
                if p[1] >= img_h:
                    p[1] = img_h

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)

            idx = np.where(segmap>0)            
            y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0])+1, np.min(idx[1]), np.max(idx[1])+1
            ref_boxes.append([x_min,y_min,x_max,y_max])
            boxes.append(box)
        crops=[]
        for box in boxes:
            # size filter for small instance
            w, h = (
                int(np.linalg.norm(box[0] - box[1]) + 1),
                int(np.linalg.norm(box[1] - box[2]) + 1),
            )
            # if w < 10 or h < 10:
            #     continue

            # warp image
            tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            M = cv2.getPerspectiveTransform(box, tar)
            crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)
            crops.append(crop)
        ref_boxes,crops=zip(*sorted(zip(ref_boxes,crops),key=lambda x: x[0][1]))
        return list(ref_boxes),list(crops)

class Locator(BaseDetector):
    def __init__(self, model_weights, img_dim=(1024,1024,3), data_channel=2, backbone="densenet121",use_cpu=True):
        super().__init__(model_weights, img_dim, data_channel, backbone=backbone,use_cpu=use_cpu)
        LOG_INFO("Loaded Detection Model,Locator",mcolor="green")
    
    def crop(self,img,debug=False):
        '''
        detects regions from an image
        args:
            img  : rgb image
        '''
        try:
            bw,bh,d=img.shape
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
            seg=softmax(pred,axis=-1)
            seg =np.argmax(seg,axis=-1)
            seg=seg.astype("uint8")
            seg=cv2.resize(seg,(org_w,org_h))

            if cfg is not None:
                if cfg["pad"]=="height":
                    seg=seg[:cfg["dim"],:]
                    seg=seg[:,:bw]
                    
                else:
                    seg=seg[:,:cfg["dim"]]
                    seg=seg[:bh,:]
            if debug:
                plt.imshow(seg)
                plt.show()        
            y_min,y_max,x_min,x_max=locateData(seg,0)
            return y_max
        except Exception as e:
            return None