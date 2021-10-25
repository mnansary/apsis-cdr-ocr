#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#-------------------------
# imports
#-------------------------
import cv2
import math
from .utils import *
import pandas as pd
import matplotlib.pyplot as plt
from .detector import CRAFT
from .robust_scanner import RobustScanner
#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,model_dir):
        '''
            Instantiates an ocr model:
            args:
                model_dir               :   path of the model weights
        '''
        
        
        dummy_path=os.path.join(os.getcwd(),"image.jpg")
        dummy_img=cv2.imread(dummy_path)
        dummy_boxes=[]
        
        # detector weight loading and initialization
        try:
            craft_weights=os.path.join(model_dir,'det',"craft.h5")
            self.det=CRAFT(craft_weights)
            
            LOG_INFO("Detector Loaded")    
            dummy_boxes=self.det.detect(dummy_img)
            if len(dummy_boxes)>0:
                LOG_INFO("Detector Initialized")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")

        # recognizer weight loading and initialization
        try:
            self.rec=RobustScanner(model_dir)
            LOG_INFO("Recognizer Loaded")
            texts=self.rec.recognize(dummy_img,dummy_boxes,
                                    batch_size=32,
                                    infer_len=10)
            if len(texts)>0:
                LOG_INFO("Recognizer Initialized")

        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        
        
        

    
    def detect_boxes(self,img,det_thresh=0.4,text_thresh=0.7,debug=False):
        '''
            detection wrapper
            args:
                img         : the np.array format image to run detction on
                det_thresh  : detection threshold to use
                text_thresh : threshold for text data
            returns:
                boxes   :   returns boxes that contains text region
        '''
        boxes=self.det.detect(img,det_thresh=det_thresh,text_thresh=text_thresh,debug=debug)
        return boxes
    
    
    

    def extract(self,img,batch_size=32,debug=False):
        '''
            predict based on datatype
            args:
                img                 :   image to infer on
                batch_size          :   batch size for inference
        '''
        # process if path is provided
        if type(img)==str:
            img=cv2.imread(img)
        # dims
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # boxes
        text_boxes=self.detect_boxes(img,debug=debug)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # recognition
        texts=self.rec.recognize(img,text_boxes,batch_size=batch_size,infer_len=20)
                
        return pd.DataFrame({"location":text_boxes,"text":texts})              


