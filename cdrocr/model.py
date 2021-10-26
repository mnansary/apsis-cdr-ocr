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
from .detector import CRAFT,Locator
from .robust_scanner import RobustScanner
from paddleocr import PaddleOCR
import copy
#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,model_dir):
        '''
            Instantiates an ocr model:
            args:
                model_dir               :   path of the model weights
            TODO:
                craft                   :   Pipeline
        '''
        # locator weight loading and initialization
        try:
            loc_weights=os.path.join(model_dir,'det',"cdr_seg.h5")
            self.loc=Locator(loc_weights)
            LOG_INFO("Locator Loaded")    
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")


        # recognizer weight loading and initialization
        try:
            self.rec=RobustScanner(model_dir)
            LOG_INFO("Recognizer Loaded")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        # db_net
        self.dbdet = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False) 
 
        # detector weight loading and initialization
        try:
            craft_weights=os.path.join(model_dir,'det',"craft.h5")
            self.det=CRAFT(craft_weights)        
            LOG_INFO("Detector Loaded")    
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")

        
        

    
    def getCrops(self,img,db_only=True,debug=False):
        '''
            detection wrapper
        '''
        base_img=np.copy(img)
        data=self.loc.crop(img,debug=debug)
        if data is None:
            return None
        else:
            try:
                img=data[0]
                reg_h=data[1]
                # resize reference
                h,w,_=img.shape
                #base_img[reg_h:,:]=(255,255,255)
                
                ref=cv2.resize(img,(w//2,h//2))
                dt_boxes,_= self.dbdet.text_detector(ref)
                dt_boxes=sorted_boxes(dt_boxes)
                # restore
                dt_boxes=dt_boxes*2
                # store crops
                crops,hs,ws=[],[],[]
                for bno in range(len(dt_boxes)):
                    tmp_box = copy.deepcopy(dt_boxes[bno])
                    img_crop = get_rotate_crop_image(img,tmp_box)
                    h,w,d=img_crop.shape
                    hs.append(h)
                    ws.append(w)
                    crops.append(img_crop)
                # number for reference
                num_idx=ws.index(max(ws))
                number=crops[num_idx]
                hf,_,_=number.shape
                nbox=dt_boxes[num_idx]
                x_max=int(max(nbox[:,0]))
                x_min=int(min(nbox[:,0]))
                y_max=int(max(nbox[:,1]))
                y_min=int(min(nbox[:,1]))                    
                #base_img[y_min:y_max,x_min:x_max]=(255,255,255)
                # collect boxes to filter
                data=[]
                boxes=[]
                for bno in range(len(dt_boxes)):
                    crop=crops[bno]
                    h,w,d=crop.shape
                    if h>hf/2:
                        data.append(crop)
                        box=dt_boxes[bno]
                        x_max=int(max(box[:,0]))
                        x_min=int(min(box[:,0]))
                        y_max=int(max(box[:,1]))
                        y_min=int(min(box[:,1]))                    
                        box=[x_min,y_min,x_max,y_max]
                        boxes.append(box)
                age=data[-2]
                names=data[:-2]
                img_list=[]
                img_list+=[number]+[age]+names
                if db_only:
                    return img_list
            except Exception as e:
                return None
        
    

    def extract(self,img,db_only=True,batch_size=32,debug=False):
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
        # img_list
        imgs=self.getCrops(self,img,db_only=db_only)        
        if imgs is None:
            return None
        else:
            texts=self.rec.recognize(None,None,image_list=imgs)
            number=texts[0]
            age=texts[1]
            name=" ".join(texts[2:])
        return [name,age,number]              


