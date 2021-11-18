#-*- coding: utf-8 -*-
"""
@author:Mobassir Hossain,MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from os import name

#-------------------------
# imports
#-------------------------
import onnxruntime as ort
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
#-------------------------
# model
#------------------------
class YOLO(object):
    def __init__(self,
                model_weights,
                providers=['CPUExecutionProvider'],
                img_dim=(512,512),
                graph_input="images"):
        self.img_dim=img_dim
        self.graph_input=graph_input
        self.model = ort.InferenceSession(model_weights, providers=providers)
    
    def process(self,img):
        pass