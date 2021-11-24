#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from os import name

#-------------------------
# imports
#-------------------------
import cv2
import math
import tensorflow as tf
from scipy.sparse import base
from .utils import *
import pandas as pd
import matplotlib.pyplot as plt
from .yolo import YOLO
from .robust_scanner import RobustScanner
import copy
#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,model_dir,use_gpu=False):
        '''
            Instantiates an ocr model:
            args:
                model_dir               :   path of the model weights
            TODO:
                craft                   :   Pipeline
        '''
        if use_gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
        # detector weight loading and initialization
        try:
            craft_weights=os.path.join(model_dir,'yolo',"yolo.onnx")
            self.det=YOLO(craft_weights)        
            LOG_INFO("Detector Loaded")    
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")

        
        # recognizer weight loading and initialization
        try:
            self.rec=RobustScanner(model_dir,"base")
            self.numrec=RobustScanner(model_dir,"num",pos_max=15)
            LOG_INFO("Recognizer Loaded")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        
        
            
    
    def extract(self,img,batch_size=32,debug=False):
        '''
            predict based on datatype
            args:
                img                 :   image to infer on
                batch_size          :   batch size for inference
        '''
        try:
            
            nums=["0","1","2","3","4","5","6","7","8","9","০","১","২","৩","৪","৫","৬","৭","৮","৯"]
            # detect
            crops=self.det.process(img,debug=debug)
            if debug:
                print(len(crops))
                for crop in crops:
                    plt.imshow(crop)
                    plt.show()

            if len(crops)<1:
                return {"name":'',"age":'',"number":'',"verdict":"low resolution image: could not detect"}
            else:
                
                if len(crops)>0 and len(crops)<=2:
                    name=''
                    age=''
                    number=''
                    texts=self.rec.recognize(None,None,image_list=crops,batch_size=batch_size)
                    for text in texts:
                        if len(text)==2 and text[0] in nums and text[1] in nums:
                            age=text
                        if len(text)>2 and text[0] in nums and text[-1] in nums:
                            number=text
                        else:
                            name=text

                    return {"name":name,"age":age,"number":number,"verdict":"low resolution image: could not detect properly/missing data in original image.probable results"}
                
                elif len(crops)==3:
                    verdict=''
                    name=crops[0]
                    age=crops[1]
                    mobile=crops[2]

                    # name
                    name=self.rec.recognize(None,None,image_list=[name],batch_size=batch_size)
                    name="".join(name)
                    # age and num
                    texts=self.numrec.recognize(None,None,image_list=[age,mobile],batch_size=batch_size,infer_len=12)
                    age=texts[0]
                    mobile=texts[1]
                    return {"name":name,"age":age,"number":mobile,"verdict":verdict}
        except Exception as e:
            return {"name":'',"age":'',"number":'',"verdict":"low resolution image: could not detect"}    

