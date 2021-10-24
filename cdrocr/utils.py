#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt 
import random
import base64
from io import BytesIO

#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr
#---------------------------------------------------------------
def locateData(img,val):
    '''
        locates img data based on specific value threshold
    '''
    idx=np.where(img>val)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    return y_min,y_max,x_min,x_max
#---------------------------------------------------------------
def remove_shadows(image: np.ndarray):
    # split the image channels into b,g,r
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    rgb_planes = [b,g,r]

    # iniatialising the final shadow free normalised image list for planes
    result_norm_planes = []

    # removing the shadows in individual planes
    for plane in rgb_planes:
        # dialting the image to spead the text to the background
        dilated_image = cv2.dilate(plane, np.ones((7,7), np.uint8))
        
        # blurring the image to get the backround image
        bg_image = cv2.medianBlur(dilated_image, 21)

        # subtracting the plane-background from the image-plane
        diff_image = 255 - cv2.absdiff(plane, bg_image)

        # normalisng the plane
        norm_image = cv2.normalize(diff_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # appending the plane to the final planes list
        result_norm_planes.append(norm_image)

    # merging the shadow-free planes into one image
    normalised_image = cv2.merge(result_norm_planes)

    # returning the normalised image
    return normalised_image
#---------------------------------------------------------------
def threshold_image(img,blur):
    '''
        threshold an image
    '''
    assert len(img.shape)==3
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold
    if blur:
        img = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img
#---------------------------------------------------------------
def cleanImage(img,remove_shadow=True,blur=True):
    '''
        cleans an image 
    '''
    # text binary
    if remove_shadow:
        img=remove_shadows(img)
    img=threshold_image(img,blur=blur)
    # remove noise
    img=cv2.merge((img,img,img))
    img= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return img
#---------------------------------------------------------------
def enhanceImage(img):
    '''
        enhances an image based on contrast
    '''
    img=Image.fromarray(img)
    # color
    col_enhancer = ImageEnhance.Color(img)
    img= col_enhancer.enhance(5)
    # contrast
    con_enhancer = ImageEnhance.Contrast(img)
    img= con_enhancer.enhance(5)
    img=np.array(img)
    return img
#--------------


#---------------------------------------------------------------
# detection utils
#---------------------------------------------------------------
def padDetectionImage(img):
    cfg={}
        
    h,w,d=img.shape
    if h>w:
        # pad widths
        pad_width =h-w
        # pads
        pad =np.ones((h,pad_width,d))*255
        # pad
        img =np.concatenate([img,pad],axis=1)
        # cfg
        cfg["pad"]="width"
        cfg["dim"]=w
    
    elif w>h:
        # pad height
        pad_height =w-h
        # pads
        pad =np.ones((pad_height,w,d))*255
        # pad
        img =np.concatenate([img,pad],axis=0)
        # cfg
        cfg["pad"]="height"
        cfg["dim"]=h
    else:
        cfg=None
    return img.astype("uint8"),cfg
   
#---------------------------------------------------------------
#  segment utils
#---------------------------------------------------------------
def order_points(pts):
    '''
        order the points a rectangle
    '''
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
#---------------------------------------------------------------
def four_point_transform(image, pts):
    '''
        4 point warping    
    '''
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
#---------------------------------------------------------------
def four_cords_crop_img(img):
    '''        
    @function author:                 
            Algo:                    
                - threshold                    
                - find max-area contour                    
                - convex hull of max-area contour                    
                - simplification of 4 point via arc-len        
            args:            
                input         = img < Image >                        
            output:               
                four_cords = [x1,y1, x2,y2, x3,y3, x4,y4]  < 1D list >               
    '''    
    # threshold image    
    ret,thresh = cv2.threshold(img,0,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest countour (c) by the area        
        c = max(contours, key = cv2.contourArea)
    # convexHull of max-area contour    
    hull = cv2.convexHull(c)
    # simplify contours
    for i in range(15):
        epsilon = (i+1)*0.01*cv2.arcLength(hull,True)
        approx = cv2.approxPolyDP(hull,epsilon,True)
        if approx.shape[-1]==4:
            break

    #epsilon = 0.1*cv2.arcLength(c,True)
    #approx = cv2.approxPolyDP(c,epsilon,True)
        
    approx=np.reshape(approx,(approx.shape[0],approx.shape[-1]))
    approx=approx.astype("float32")
    pts=np.array([approx[0],approx[3],approx[2],approx[1]])
    return pts
#---------------------------------------------------------------
# recognition utils
#---------------------------------------------------------------
def padData(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8") 
#---------------------------------------------------------------
def padWords(img,dim,ptype="central",pvalue=255,scope_pad=50):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padData(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padData(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w+scope_pad
        if mask>img_width:
            mask=img_width
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#---------------------------------------------------------------
def processCleanWord(img):
    '''
        processes a clean word
    '''
    img=threshold_image(img,blur=False)
    y_min,y_max,x_min,x_max=locateData(img,0)
    img=img[y_min:y_max,x_min:x_max]
    img=cv2.merge((img,img,img))
    return img
#------------------------------------
# region-utils 
#-------------------------------------
def intersection(boxA, boxB):
    # boxA=ref
    # boxB=sig
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    x_min,y_min,x_max,y_max=boxB
    selfArea  = abs((y_max-y_min)*(x_max-x_min))
    return interArea/selfArea
#---------------------------------------------------------------
def localize_box(box,region_boxes):
    '''
        lambda localization
    '''
    max_ival=0
    box_id=None
    for idx,region_box in enumerate(region_boxes):
        ival=intersection(region_box,box)
        if ival==1:
            return idx
        if ival>max_ival:
            max_ival=ival
            box_id=idx
    if max_ival==0:
        return None
    return box_id

#----------------------------------------
# display utils
#----------------------------------------
def display_data(info,img,cv_color=True):
    if type(img)=="str":
        img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else:
        if cv_color:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    LOG_INFO("------------------------------------------------")
    LOG_INFO(f"{info}")
    LOG_INFO("------------------------------------------------")
    plt.imshow(img)
    plt.show()

#-----------------------
# response utils
#-----------------------

def img2base64(img_detections):
    '''
        creates proper json response
    '''
    # Convert numpy array To PIL image
    pil_detection_img = Image.fromarray(img_detections)

    # Convert PIL image to bytes
    buffered_detection = BytesIO()

    # Save Buffered Bytes
    pil_detection_img.save(buffered_detection, format='PNG')

    # Base 64 encode bytes data
    # result : bytes
    base64_detection = base64.b64encode(buffered_detection.getvalue())

    # Decode this bytes to text
    # result : string (utf-8)
    base64_detection = base64_detection.decode('utf-8')
    return base64_detection

#--------------------------------------
# text checking utils: TODO
#--------------------------------------
# correct numbers
'''
if "o" in response["ID No."]:
    response["ID No."]=response["ID No."].replace("o","0")
if "s" in response["ID No."]:
response["ID No."]=response["ID No."].replace("s","5")
if "z" in response["ID No."]:
response["ID No."]=response["ID No."].replace("z","2")
if "i" in response["ID No."]:
response["ID No."]=response["ID No."].replace("i","1")
if "l" in response["ID No."]:
response["ID No."]=response["ID No."].replace("l","1")
'''