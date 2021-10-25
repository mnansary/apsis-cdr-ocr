#!/bin/sh
base_path="/home/apsisdev/ansary/DATASETS/GVU/"
bw_ref="/home/apsisdev/ansary/DATASETS/RAW/bangla_writing/converted/converted/"
bh_ref="/home/apsisdev/ansary/DATASETS/RAW/BN-HTR/"
bs_ref="/home/apsisdev/ansary/DATASETS/RAW/BanglaC/README.txt"
iit_path="/home/apsisdev/Rezwan/cvit_iiit-indic/"
save_path="/home/apsisdev/ansary/DATASETS/APSIS/CDR/"
nb_dir="/home/apsisdev/ansary/DATASETS/APSIS/Recognition/source/base/"
#iit_path="/media/ansary/DriveData/Work/APSIS/datasets/__raw__/bengal/iiit-indic/"
#base_path="/media/ansary/DriveData/Work/APSIS/datasets/GVU/"
#-----------------------------------------------------------------------------------------------
data_dir="${base_path}data/"
ds_path="${save_path}datasets/"
bw_ds="${ds_path}bw/"
bh_ds="${ds_path}bh/"
bs_ds="${ds_path}bs/"
bn_pr_ds="${ds_path}bangla_printed/"
bn_hr_ds="${ds_path}bn.synth/"
nb_ds="${ds_path}nb.synth/"

iit_bn_ref="${iit_path}bn/vocab.txt"
iit_bn_ds="${ds_path}iit.bn/"

#-----------------------------------bangla-----------------------------------------------
#-----------------------------------natrual---------------------------------------------
#python datasets/bangla_writing.py $bw_ref $ds_path
#python datagen.py $bw_ds 
#python datasets/boise_state.py $bs_ref $ds_path
#python datagen.py $bs_ds 
#python datasets/bn_htr.py $bh_ref $ds_path
#python datagen.py $bh_ds 
#python datasets/iit_indic.py $iit_bn_ref $ds_path
#python datagen.py $iit_bn_ds 
#python create_recs.py $bn_hr_ds "bn.synth"

#-----------------------------------natrual---------------------------------------------

#-----------------------------------synthetic------------------------------------------
#python datagen_synth.py $data_dir "bangla" "printed" $ds_path --num_samples 1000000
#python datagen_synth.py $data_dir "bangla" "handwritten" $ds_path --num_samples 500000
python datagen_num.py $nb_dir $ds_path --num_samples 10

#-----------------------------------synthetic------------------------------------------
#-----------------------------------bangla-----------------------------------------------


echo succeeded