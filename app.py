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
            number,age,name,sec_name=data
            # check number
            for n in number:
                rec_num=[]
                if n not in EN_NUMS+BN_NUMS:
                    number_ret=False
                else:
                    rec_num.append(n)
            
            if number_ret:
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
                if len(age)!=2:
                    response["age"]=f"Full Age not found:Recognized Age:{''.join(rec_age)}"
                else:
                    response["age"]=age
            if sec_name.strip() and name.strip()!=sec_name.strip():
                response["name"]=f"Multiple Name Candidates Found- 1.{name} 2.{sec_name} "
            else:
                response["Name"]=f"{name}"
            
            
            
        print(response)
        return jsonify(response)
    return None


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
