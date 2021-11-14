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

from scipy.sparse import base
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

            
    def getCrops(self,img,debug=False,h_thresh=0.8):
        '''
            detection wrapper
        '''
        base_img=np.copy(img)
        data=self.loc.crop(img,debug=debug)
        if data is None:
            return "LOG:locator failed to locate name,age and number region: more image training data needed"
        else:
            try:
                reg_h=data
                # dbnet
                dt_boxes,_= self.dbdet.text_detector(img) 
                dt_boxes=sorted_boxes(dt_boxes)
                # store crops
                crops=[]
                crop_boxes=[]
                for bno in range(len(dt_boxes)):
                    tmp_box = copy.deepcopy(dt_boxes[bno])
                    y_min=int(min(tmp_box[:,1]))    
                    # filter based on region                
                    if y_min<reg_h:
                        img_crop = get_rotate_crop_image(img,tmp_box)
                        crops.append(img_crop)
                        crop_boxes.append(tmp_box)
                        if debug:
                            plt.imshow(img_crop)
                            plt.show()
                
                
                # number for reference
                number=crops[-1]
                hf,_,_=number.shape
                if debug:
                    plt.imshow(number)
                    plt.show()
                # filter base_img
                y_reg=int(min(crop_boxes[-1][:,1]))    
                base_img[y_reg:,:]=(255,255,255)
                if debug:
                    plt.imshow(base_img)
                    plt.show()

                # craft det
                boxes=self.det.detect(base_img,debug=debug)
                ref_boxes=[]
                words=[]
                for box in boxes:
                    x1,y1,x2,y2=box
                    hw=y2-y1
                    # filter
                    if hw/hf>h_thresh: 
                        word=base_img[y1:y2,x1:x2]
                        ref_boxes.append(box)
                        words.append(word)
                        if debug:
                            plt.imshow(word)
                            plt.show()
                # last one should be age
                ref_boxes,words=zip(*sorted(zip(ref_boxes,words),key=lambda x: x[0][1]))
                age=words[-1]

                
                # the rest are name
                ref_boxes,words=zip(*sorted(zip(ref_boxes[:-1],words[:-1]),key=lambda x: x[0][0]))

                if debug:
                    print("Data:")
                    plt.imshow(age)
                    plt.show()
                    for word in words:
                        plt.imshow(word)
                        plt.show()
                
                img_list=[]
                img_list+=[number]+[age]+list(words)
                if debug:
                    for img in img_list:
                        plt.imshow(img)
                        plt.show()
                return img_list
                
            except Exception as e:
                print(e)
                return "LOG:Problem while cropping words: more image training data needed"
        
    

    def extract(self,img,batch_size=32,debug=False,h_thresh=0.8):
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
        data=self.getCrops(img,debug=debug,h_thresh=h_thresh)        
        if type(data)==str:
            return data
        else:
            imgs=data
            texts=self.rec.recognize(None,None,image_list=imgs,batch_size=batch_size)
            number=texts[0]
            age=texts[1]
            name=" ".join(texts[2:])
            
            
            
        return [number,age,name]              


