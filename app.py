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
EN_NUMS=["0","1","2","3","4","5","6","7","8","9"]
BN_NUMS=['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



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
        if type(data)==str:
            response["error"]=data
        else:
            number_ret=True
            age_ret=True
            number,age,name=data
            # check number
            for n in number:
                rec_num=[]
                if n not in EN_NUMS+BN_NUMS:
                    number_ret=False
                else:
                    rec_num.append(n)
            
            if number_ret:
                assert len(number)==len(rec_num)
                if len(number)!=11:
                    response["number"]=f"Full Mobile Number not found:Recognized numbers:{''.join(rec_num)}"
                else:
                    response["number"]=number

            # check age
            for n in age:
                rec_age=[]
                if n not in EN_NUMS+BN_NUMS:
                    age_ret=False
                else:
                    rec_age.append(n)
            
            if age_ret:
                assert len(age)==len(rec_age)
                if len(number)!=2:
                    response["age"]=f"Full Age not found:Recognized numbers:{''.join(rec_num)}"
                else:
                    response["age"]=age
            # chek name
            response["Name"]=name
            
            
        print(response)
        return jsonify(response)
    return None


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
