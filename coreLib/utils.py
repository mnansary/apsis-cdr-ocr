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
from tqdm import tqdm
import random
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

def randColor(col=True):
    '''
        generates random color
    '''
    if col:
        return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    else:
        d=random.randint(0,64)
        return (d,d,d)

def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match
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

def removeShadow(img):
    '''
        removes shadows
    '''
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def padAllAround(img,pad_dim,pad_val,pad_single=None):
    '''
        pads all around the image
    '''
    if pad_single is None:
        h,w=img.shape
        # pads
        left_pad =np.ones((h,pad_dim))*pad_val
        right_pad=np.ones((h,pad_dim))*pad_val
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
        # shape
        h,w=img.shape
        top_pad =np.ones((pad_dim,w))*pad_val
        bot_pad=np.ones((pad_dim,w))*pad_val
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    elif pad_single=="tb":
        # shape
        h,w=img.shape
        top_pad =np.ones((pad_dim,w))*pad_val
        bot_pad=np.ones((pad_dim,w))*pad_val
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    else:
        h,w=img.shape
        # pads
        left_pad =np.ones((h,pad_dim))*pad_val
        right_pad=np.ones((h,pad_dim))*pad_val
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
    return img
#---------------------------------------------------------------
# parsing utils
#---------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
#---------------------------------------------------------------
# text utils
#---------------------------------------------------------------
class baselang:
    vowel_diacritics       =    ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
    consonant_diacritics   =    ['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
    modifiers              =    []
    connector              =    '্'
    
class GraphemeParser(object):
    def __init__(self,language=None):
        '''
            initializes a grapheme parser for a given language
            args:
                language  :   a class that contains list of:
                                1. vowel_diacritics 
                                2. consonant_diacritics
                                3. modifiers
                                and 
                                4. connector 
        '''
        if language==None:
            language=baselang
        # assignment
        self.vds=language.vowel_diacritics 
        self.cds=language.consonant_diacritics
        self.mds=language.modifiers
        self.connector=language.connector
        # error check -- type
        assert type(self.vds)==list,"Vowel Diacritics Is not a list"
        assert type(self.cds)==list,"Consonant Diacritics Is not a list"
        assert type(self.mds)==list,"Modifiers Is not a list"
        assert type(self.connector)==str,"Connector Is not a string"
    
    def get_root_from_decomp(self,decomp):
        '''
            creates grapheme root based list 
        '''
        add=0
        if self.connector in decomp:   
            c_idxs = [i for i, x in enumerate(decomp) if x == self.connector]
            # component wise index map    
            comps=[[cid-1,cid,cid+1] for cid in c_idxs ]
            # merge multi root
            r_decomp = []
            while len(comps)>0:
                first, *rest = comps
                first = set(first)

                lf = -1
                while len(first)>lf:
                    lf = len(first)

                    rest2 = []
                    for r in rest:
                        if len(first.intersection(set(r)))>0:
                            first |= set(r)
                        else:
                            rest2.append(r)     
                    rest = rest2

                r_decomp.append(sorted(list(first)))
                comps = rest
            # add    
            combs=[]
            for ridx in r_decomp:
                comb=''
                for i in ridx:
                    comb+=decomp[i]
                combs.append(comb)
                for i in ridx:
                    decomp[i]=comb
                    
            # new root based decomp
            new_decomp=[]
            for i in range(len(decomp)-1):
                if decomp[i] not in combs:
                    new_decomp.append(decomp[i])
                else:
                    if decomp[i]!=decomp[i+1]:
                        new_decomp.append(decomp[i])

            new_decomp.append(decomp[-1])#+add*connector
            
            return new_decomp
        else:
            return decomp

    def get_graphemes_from_decomp(self,decomp):
        '''
        create graphemes from decomp
        '''
        graphemes=[]
        idxs=[]
        for idx,d in enumerate(decomp):
            if d not in self.vds+self.mds:
                idxs.append(idx)
        idxs.append(len(decomp))
        for i in range(len(idxs)-1):
            sub=decomp[idxs[i]:idxs[i+1]]
            grapheme=''
            for s in sub:
                grapheme+=s
            graphemes.append(grapheme)
        return graphemes

    def process(self,word,return_graphemes=False):
        '''
            processes a word for creating:
            if return_graphemes=False (default):
                * components
            else:                 
                * grapheme 
        '''
        try:
            decomp=[ch for ch in word]
            decomp=self.get_root_from_decomp(decomp)
            if return_graphemes:
                return self.get_graphemes_from_decomp(decomp)
            else:
                components=[]
                for comp in decomp:
                    if comp in self.vds+self.mds:
                        components.append(comp)
                    else:
                        cd_val=None
                        for cd in self.cds:
                            if cd in comp:
                                comp=comp.replace(cd,"")
                                cd_val=cd
                        components.append(comp)
                        if cd_val is not None:
                            components.append(cd_val)
                return components
        except Exception as e:
            LOG_INFO(e)
            LOG_INFO(word)                        