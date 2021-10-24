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
from .classification import Classifier
from deepface import DeepFace
from .data import card
from .locator import Locator
from paddleocr import PaddleOCR
#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,model_dir,use_facematcher=False):
        '''
            Instantiates an ocr model:
            args:
                model_dir               :   path of the model weights
                use_facematcher         :   loading facematcher
                use_robustScanner       :   loading robust-scanner
        '''
        
        # nid dummy image
        dummy_path=os.path.join(os.getcwd(),"tests","det.png")
        dummy_img=cv2.imread(dummy_path)
        dummy_boxes=[]
        for _,v in card.nid.front.box_dict.items():
            dummy_boxes.append(v)
        # classifier weight loading and initialization
        try:
            ext_weights=os.path.join(model_dir,"cls","classifier.h5")
            self.classifier=Classifier(ext_weights)
            LOG_INFO("Classifier Loaded")    
            card_type=self.classifier.process(cv2.cvtColor(dummy_img,cv2.COLOR_BGR2RGB))
            if card_type=="nid":
                LOG_INFO("Classifier Initialized")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        
        # locator weight loading and initialization
        try:
            self.locator=Locator(weights_path=model_dir)
            LOG_INFO("Locator Loaded")    
            ref=self.locator.process(cv2.cvtColor(dummy_img,cv2.COLOR_BGR2RGB))
            LOG_INFO("Locator Initialized")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")

        # detector weight loading and initialization
        try:
            craft_weights=os.path.join(model_dir,'det',"craft.h5")
            self.det=CRAFT(craft_weights)
            
            LOG_INFO("Detector Loaded")    
            boxes=self.det.detect(dummy_img)
            if len(boxes)>0:
                LOG_INFO("Detector Initialized")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
            



        # recognizer weight loading and initialization
        self.engocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False) 
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
        
        # facematcher loading and initialization
        if use_facematcher:
            try:
                obj=DeepFace.verify(dummy_path,dummy_path,model_name = 'ArcFace', detector_backend = 'retinaface')
                if obj['verified']:
                    LOG_INFO("Face Matcher Initialized")
            except Exception as e:
                LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")

        

    def facematcher(self,src_img,dest_img,return_dict=False):
        '''
            matches two given faces:
            args:
                src_img     :   the given id image (np.array)
                dest_img    :   the selfie/taken image to match (np.array)
                return_dict :   if set to True returns a dictionary of {match,similiarity} 
        '''
        src=os.path.join(os.getcwd(),"tests","src.png")
        dst=os.path.join(os.getcwd(),"tests","dst.png")
        cv2.imwrite(src,src_img)
        cv2.imwrite(dst,dest_img)
        obj=DeepFace.verify(src,dst,model_name = 'ArcFace', detector_backend = 'retinaface')
        
        match=obj["verified"]
        similiarity_value=round(obj['distance']*100,2)
        
        if match:            
            LOG_INFO(f"FOUND MATCH:  Similiarity ={similiarity_value}%")
        else:
            LOG_INFO(f"MATCH NOT FOUND:  Similiarity ={similiarity_value}%")
        if return_dict:
            return {"match":match,"similiarity":similiarity_value}

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
    
    def process_boxes(self,text_boxes,region_dict,rx,ry):
        '''
            keeps relevant boxes with respect to region
            args:
                text_boxes  :  detected text boxes by the detector
                region_dict :  key,value pair dictionary of region_bbox and field info 
                               => {"field_name":[x_min,y_min,x_max,y_max]}
                rx          :  x dim ratio
                ry          :  y dim ratio
        '''
        # extract region boxes
        region_boxes=[]
        region_fields=[]
        for k,v in region_dict.items():
            region_fields.append(k)
            region_boxes.append(v)
        # ref boxes
        ref_boxes=[]
        for box in text_boxes:
            x1,y1,x2,y2=box
            ref_boxes.append([int(math.ceil(x1*rx)),
                              int(math.ceil(y1*ry)),
                              int(math.ceil(x2*rx)),
                              int(math.ceil(y2*ry))])
        # sort boxed
        data=pd.DataFrame({"box":text_boxes,"ref_box":ref_boxes})
        # detect field
        data["field"]=data.ref_box.apply(lambda x:localize_box(x,region_boxes))
        data.dropna(inplace=True) 
        data["field"]=data["field"].apply(lambda x:region_fields[int(x)])
        box_dict={}
        df_box=[]
        df_field=[]
        for field in data.field.unique():
            df=data.loc[data.field==field]
            boxes=df.box.tolist()
            boxes=sorted(boxes, key=lambda x: x[0])
            box_dict[field]=boxes

            for box in boxes:
                df_box.append(box)
                df_field.append(field)
        
        df=pd.DataFrame({"box":df_box,"field":df_field})
        return box_dict,df 

    

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
            org=np.copy(img)
           
        # dims
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        card_type=self.classifier.process(img)
        
        if card_type=="nid": 
            src=card.nid.front
        else: 
            src=card.smart.front
            
        # locator
        img=self.locator.process(org)
        if debug:
            plt.imshow(img)
            plt.show()
        # face and sign
        ref=cv2.resize(img,(card.width,card.height))
        
        x1,y1,x2,y2=src.face
        face=ref[y1:y2,x1:x2]
        x1,y1,x2,y2=src.sign
        sign=ref[y1:y2,x1:x2]
        if debug:
            plt.imshow(ref)
            plt.show()
            plt.imshow(face)
            plt.show()
            plt.imshow(sign)
            plt.show()
        
        
        
        # ratio
        ho,wo,d=img.shape
        hr,wr,d=ref.shape
        rx=wr/wo
        ry=hr/ho
        # boxes
        text_boxes=self.detect_boxes(img,debug=debug)
        box_dict,df=self.process_boxes(text_boxes,src.box_dict,rx,ry)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # recognition
        eng_keys=["English Name","Date of Birth","ID No."]
        texts=[]
        for k,v in box_dict.items():
            if k in eng_keys:
                for box in v:
                    # crop    
                    x_min,y_min,x_max,y_max=box
                    word=img[y_min:y_max,x_min:x_max] 
                    result = self.engocr.ocr(word, cls=True,det=False,rec=True)
                    # find data
                    for line in result:
                        texts.append(line[0])
                
            else:
                texts+=self.rec.recognize(img,v,batch_size=batch_size,infer_len=20)
                
        response={}
        response["card_type"]=card_type
        df["text"]=texts
        for field in df.field.unique():
            tdf=df.loc[df.field==field]
            if field=="ID No." and card_type=="nid":
                response[field]="".join(tdf.text.tolist())
            else:
                response[field]=" ".join(tdf.text.tolist())
            if field in eng_keys and card_type=="smart":
                response[field]=response[field].upper()

            
        response["image"]=img2base64(face)
        response["sign"]=img2base64(sign)
        
        return response                  


