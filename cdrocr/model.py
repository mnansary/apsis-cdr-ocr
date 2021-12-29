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
from paddleocr import PaddleOCR
import copy
from difflib import get_close_matches
#-------------------------
# class
#------------------------
brand_list  =  ["B&H","JPGL","Capstan","Marise",
                "Marlboro","Pallmall","Star","Sheikh",
                "LD","Castle","Rally","Winston",
                "Hollywood","Pilot","Real",
                "Navy","Derby","Royals"]

class OCR(object):
    def __init__(self,model_dir):
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
            yolo_weights=os.path.join(model_dir,'yolo',"best.onnx")
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
        
        # engocr
        self.engocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False)

        
            
    
    def extract(self,img,batch_size=32,debug=False):
        '''
            predict based on datatype
            args:
                img                 :   image to infer on
                batch_size          :   batch size for inference
        '''
        try:
        
    
            data={}

            if type(img)==str:
                img=cv2.imread(img)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            base=np.copy(img)
            img,_=padDetectionImage(img)
            
            # yolo
            res,ref_boxes=self.yolo.process(img)
            if debug:
                for k,v in res.items():
                    print(k)
                    if v is not None:
                        plt.imshow(v["crop"])
                        plt.show()

            # age and number
            num_labels=['age', 'mobile']
            for num_label in num_labels:
                if res[num_label] is not None:
                    texts=self.numrec.recognize(None,None,image_list=[res[num_label]["crop"]],infer_len=12)
                    data[num_label]=texts[0]
                else:
                    data[num_label]="Not Present In Image"
            # name
            if res["name"] is None:
                data["name"]="Not Present In Image"
            else:
                text_boxes,text_crops=self.craft.detect(img,debug=debug)
                name_boxes=[]
                name_crops=[]
                for tbox,ncrop in zip(text_boxes,text_crops):
                    idx=localize_box(tbox,[res["name"]["bbox"]])
                    if idx==0:
                        name_boxes.append(tbox)
                        name_crops.append(ncrop)
                
                if len(name_boxes)>0:
                    name_boxes,name_crops=zip(*sorted(zip(name_boxes,name_crops),key=lambda x: x[0][0]))            
                    name=self.rec.recognize(None,None,image_list=list(name_crops),batch_size=batch_size)
                    name=" ".join(name)
                    data["name"]=name
                else:
                    name=self.rec.recognize(None,None,image_list=[res["name"]["bbox"]],batch_size=batch_size)
                    data["name"]=name[0]
                    
            # smoker
            if res["smoker"] is None:
                data["smoker"]="Not Present In Image"
            else:
                data["smoker"]="yes"
            
            dt_boxes,_= self.engocr.text_detector(base) 
            dt_boxes=sorted_boxes(dt_boxes)

            # brand
            if res["brand"] is None:
                data["brand"]="Not Present In Image"
            else:
                data["brand"]=""
                for bno in range(len(dt_boxes)):
                    tmp_box = copy.deepcopy(dt_boxes[bno])
                    if debug:
                        img_crop = get_rotate_crop_image(img,tmp_box)
                        plt.imshow(img_crop)
                        plt.show()
                    x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
                    y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
                    idx=localize_box([x1,y1,x2,y2],[res["brand"]["bbox"]])
                    if idx==0:
                        try:
                            img_crop = get_rotate_crop_image(img,tmp_box)
                            texts=self.engocr.ocr(img_crop,det=False)
                            brand=texts[0][0]
                            brand=get_close_matches(brand,brand_list,n=1)[0]
                            data["brand"]=brand
                        except Exception as eb:
                            data["brand"]="brand region Noisy/Low resolution.Try another image"

            return data
        except Exception as e:
            print(data)
            print(e)
            return {"verdict":"low resolution image: could not detect"}    

