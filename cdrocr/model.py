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
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # detector weight loading and initialization
        try:
            yolo_weights=os.path.join(model_dir,'yolo',"yolo.onnx")
            self.yolo=YOLO(yolo_weights)
            LOG_INFO("YOLO Loaded")
            craft_weights=os.path.join(model_dir,'det',"craft.h5")
            self.craft=CRAFT(craft_weights)
                    
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
        
        
            
    
    def extract(self,img,batch_size=32,img_dim=(512,512),debug=False):
        '''
            predict based on datatype
            args:
                img                 :   image to infer on
                batch_size          :   batch size for inference
        '''
        try:
            
            nums=["0","1","2","3","4","5","6","7","8","9","০","১","২","৩","৪","৫","৬","৭","৮","৯"]
            if type(img)==str:
                img=cv2.imread(img)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            h,w,_=img.shape
            img,_=padDetectionImage(img)
            img=cv2.resize(img,img_dim)
        
            # yolo
            ref_boxes=self.yolo.process(img)
            
            if len(ref_boxes)<1:
                return {"name":'',"age":'',"number":'',"verdict":"low resolution image: could not detect"}
            else:
                
                if len(ref_boxes)>0 and len(ref_boxes)<=2:
                    crops=[]
                    for box in ref_boxes:
                        x1,y1,x2,y2=box
                        crops.append(img[y1:y2,x1:x2])
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
                
                elif len(ref_boxes)==3:
                    text_boxes=self.craft.detect(img,debug=debug)
                    
                    crops=[]
                    for box in ref_boxes:
                        x1,y1,x2,y2=box
                        crops.append(img[y1:y2,x1:x2])
                        if debug:
                            plt.imshow(img[y1:y2,x1:x2])
                            plt.show()
                    
                    verdict=''
                    age=crops[1]
                    mobile=crops[2]
                    # age and num
                    texts=self.numrec.recognize(None,None,image_list=[age,mobile],batch_size=batch_size,infer_len=12)
                    age=texts[0]
                    mobile=texts[1]

                    # name
                    name_boxes=[]
                    for tbox in text_boxes:
                        idx=localize_box(tbox,ref_boxes)
                        if idx==0:
                            name_boxes.append(tbox)
                    name_boxes=sorted(name_boxes,key=lambda x: x[0])    

                    crops=[]
                    for box in name_boxes:
                        x1,y1,x2,y2=box
                        crops.append(img[y1:y2,x1:x2])
                        if debug:
                            plt.imshow(img[y1:y2,x1:x2])
                            plt.show()
                    

                    name=self.rec.recognize(None,None,image_list=crops,batch_size=batch_size)
                    name=" ".join(name)
                    
                    return {"name":name,"age":age,"number":mobile,"verdict":verdict}
        except Exception as e:
            return {"name":'',"age":'',"number":'',"verdict":"low resolution image: could not detect"}    

