#!/bin/bash

DIR=../models/rbf/

ENC=rbf

DATA=birdsnap
# DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
# META=ebird_meta
META=orig_meta
EVALDATA=test

DEVICE=cuda:0

LR=0.005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.1
# MAXR=1
EPOCH=30

ACT=relu
RATIO=1.0

#num_rbf_anchor_pts = 200 #[100, 200, 500]
#rbf_kernel_size = # [0.5, 1, 2, 10]
KERNELSIZE=2

for x in nabirds,ebird_meta,test  #inat_2018,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.003 0.002 0.001
    do
        python3 main.py \
            --spa_enc_type $ENC \
            --meta_type $META\
            --dataset $DATA \
            --eval_split $EVALDATA \
            --frequency_num $FREQ \
            --min_radius $MINR \
            --num_hidden_layer $LAYER \
            --hidden_dim $HIDDIM \
            --spa_f_act $ACT \
            --unsuper_lr 0.1 \
            --lr $LR \
            --model_dir $DIR \
            --num_epochs $EPOCH \
            --train_sample_ratio $RATIO \
            --device $DEVICE \
            --num_rbf_anchor_pts 200 \
            --rbf_kernel_size $KERNELSIZE
    done
done