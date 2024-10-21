#!/bin/bash

DIR=../models/rbf/

ENC=rbf

# DATA=birdsnap
# DATA=inat_2017
# DATA=inat_2018
DATA=fmow
META=ebird_meta
# META=orig_meta
EVALDATA=val

DEVICE=cuda:3

LR=0.005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.0005
MAXR=1
EPOCH=100

ACT=relu
RATIO=1.0

KERNELSIZE=2
ANCHOR=200

for x in fmow,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.005
    do
        for MINR in 0.0005
        do
            python3 main.py \
                --spa_enc_type $ENC \
                --meta_type $META\
                --dataset $DATA \
                --eval_split $EVALDATA \
                --frequency_num $FREQ \
                --max_radius $MAXR \
                --min_radius $MINR \
                --num_hidden_layer $LAYER \
                --hidden_dim $HIDDIM \
                --spa_f_act $ACT \
                --unsuper_lr 0.1 \
                --lr $LR \
                --model_dir $DIR \
                --num_epochs $EPOCH \
                --train_sample_ratio $RATIO \
                --num_rbf_anchor_pts $ANCHOR \
                --rbf_kernel_size $KERNELSIZE \
                --device $DEVICE
        done
    done
done
