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
from coreLib.synthetic import createSyntheticData
from coreLib.languages import languages
from coreLib.processing import processData
from coreLib.store import createRecords
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_dir    =   args.data_dir
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    pad_height  =   int(args.pad_height)
    num_samples =   int(args.num_samples)
    dict_max_len=   int(args.dict_max_len)
    dict_min_len=   int(args.dict_min_len)
    iden        =   args.iden
    seq_max_len =   int(args.seq_max_len)
    
    img_dim=(img_height,img_width)
    # data creation bn num
    language=languages["bangla"]
    df1,off,csv=createSyntheticData(iden=iden,
                        save_dir=save_path,
                        data_type="handwritten",
                        data_dir=data_dir,
                        language=language,
                        num_samples=num_samples,
                        dict_max_len=dict_max_len,
                        dict_min_len=dict_min_len,
                        comp_dim=img_height,
                        pad_height=pad_height,
                        use_all=False,
                        use_only_numbers=True,
                        return_df=True)
    # data creation en num
    language=languages["bangla"]
    df2,_,csv=createSyntheticData(iden=iden,
                        save_dir=save_path,
                        data_type="handwritten",
                        data_dir=data_dir,
                        language=language,
                        num_samples=num_samples,
                        dict_max_len=dict_max_len,
                        dict_min_len=dict_min_len,
                        comp_dim=img_height,
                        pad_height=pad_height,
                        use_all=False,
                        use_only_numbers=True,
                        fname_offset=off,
                        return_df=True)
    
    df=pd.concat([df1,df2],ignore_index=True)
    df.to_csv(csv,index=False)

    # processing
    df=processData(csv,language,seq_max_len,img_dim,return_df=True)
    # storing
    save_path=os.path.dirname(csv)
    save_path=create_dir(save_path,iden)
    createRecords(df,save_path)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--iden",required=False,default=None,help="identifier to identify the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    parser.add_argument("--pad_height",required=False,default=20,help ="pad height for each grapheme for alignment correction: default=20")
    parser.add_argument("--num_samples",required=False,default=50000,help ="number of samples to create when:default=50000")
    parser.add_argument("--dict_max_len",required=False,default=13,help=" the maximum length of data for randomized dictionary")
    parser.add_argument("--dict_min_len",required=False,default=1,help=" the minimum length of data for randomized dictionary")
    parser.add_argument("--seq_max_len",required=False,default=80,help=" the maximum length of data for modeling")
    args = parser.parse_args()
    main(args)