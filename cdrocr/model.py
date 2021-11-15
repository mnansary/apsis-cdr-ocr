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
from .detector import CRAFT
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
            
        # recognizer weight loading and initialization
        try:
            self.rec=RobustScanner(model_dir)
            LOG_INFO("Recognizer Loaded")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        
        # detector weight loading and initialization
        try:
            craft_weights=os.path.join(model_dir,'det',"craft.h5")
            self.det=CRAFT(craft_weights,use_cpu=False)        
            LOG_INFO("Detector Loaded")    
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
            # process if path is provided
            if type(img)==str:
                img=cv2.imread(img)
            # dims
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # detect
            ref_boxes,crops=self.det.detect(img,debug=debug)
            if len(ref_boxes)<1:
                return {"name":'',"age":'',"number":'',"verdict":"low resolution image: could not detect"}
            else:
                verdict=''
                ws=[]
                hs=[]
                for crop in crops:
                    h,w,_=crop.shape
                    ws.append(w)
                    hs.append(h)
                    if debug:
                        plt.imshow(crop)
                        plt.show()
                num_idx=np.argmax(ws)
                
                mobile=crops[num_idx]
                age=crops[num_idx-1]
                # name
                rest=crops[:num_idx-1]
                ref_boxes=ref_boxes[:len(rest)]
                ref_boxes,rest=zip(*sorted(zip(ref_boxes,rest),key=lambda x: x[0][0]))
                
                    
                crops=list(rest)+[age]+[mobile]
                texts=self.rec.recognize(None,None,image_list=crops,batch_size=batch_size)
                # data
                number=texts[-1]
                if number[0] not in nums:
                    number=''
                    verdict+="mobile number not found:low resolution/ irregular image\n"

                if texts[-2][0] in nums:
                    age=texts[-2]
                else:
                    age=''
                    verdict+="age not found:low resolution/ irregular image\n"

                if len(texts)<3:
                    name=''
                    verdict+="name not found:low resolution/ irregular image\n"
                else:
                    name=" ".join(list(texts)[:-2])
                
                if len(name)==0:
                    verdict+="name not found:low resolution/ irregular image\n"

                return {"name":name,"age":age,"number":number,"verdict":verdict}
        except Exception as e:
            return {"name":'',"age":'',"number":'',"verdict":"low resolution image: could not detect"}    

