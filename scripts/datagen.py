#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
import json
import pandas as pd 

from tqdm import tqdm
from ast import literal_eval

from coreLib.utils import *
from coreLib.processing import processData
from coreLib.store import createRecords
from coreLib.languages import vocab
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_dir    =   args.data_dir
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    iden        =   args.iden
    seq_max_len =   int(args.seq_max_len)
    
    if iden is None:
        iden=os.path.basename(os.path.dirname(data_dir))
    img_dim=(img_height,img_width)
    csv=os.path.join(data_dir,"data.csv")
    # processing
    df=processData(csv,vocab,seq_max_len,img_dim)
    # storing
    save_path=create_dir(data_dir,iden)
    LOG_INFO(save_path)
    createRecords(df,save_path)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("--iden",required=False,default=None,help="identifier to identify the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    parser.add_argument("--seq_max_len",required=False,default=40,help=" the maximum length of data for modeling")
    args = parser.parse_args()
    main(args)