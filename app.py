#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
# models
from cdrocr.model import OCR
# Define a flask app
app = Flask(__name__)

ocr=OCR("models/")
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

NUMS=["0","1","2","3","4","5","6","7","8","9",'০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,"tests",'uploads', secure_filename(f.filename))
        f.save(file_path)
        data=ocr.extract(file_path)
        response={}
        if data is None:
            response["error"]="problem reading file (segmentation problem)"
            response["recognizer"]="Word Recognition working acurately!(double check)-could be an invalid image"
            response["locator"]="Couldn't locate number"
            response["detector"]="Number split up due to resolution or improper image"
        else:
            number_ret=True
            age_ret=True
            number,age,name=data
            for n in number:
                nn=[]
                if n not in NUMS:
                    number_ret=False
                else:
                    nn.append(n)

            for n in age:
                aa=[]
                if n not in NUMS:
                    age_ret=False
                else:
                    aa.append(n)
            if len(aa)!=2:
                age_ret=False

            if number_ret:
                response["Mobile Number"]=number
            else:
                response["LOG:ocr-recognition"]=f"Working good.Number Recognition:{''.join(nn)}"    
                response["detector-error"]="Number split up due to resolution or improper image(Need more training images for location)"
            response["Name"]=name
            if age_ret:
                response["Age"]=age
            else:
                response["LOG:ocr-recognition"]=f"Working good:{age}"    
                response["locator-error"]="Age position ambogious(Need more training images for location)"

            
        print(response)
        return jsonify(response)
    return None


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
