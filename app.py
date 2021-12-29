#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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

ocr=None


@app.route('/', methods=['GET'])
def index():
    global ocr
    ocr=OCR("models/")
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath,"tests",'uploads', secure_filename(f.filename))
            f.save(file_path)
            data=ocr.extract(file_path)
            print(data)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error":"upload failed"})
    
    return jsonify({"error":"upload failed"})


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")
