#!/bin/bash

DIR=../models/wrap/

ENC=wrap

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
MINR=0.02
MAXR=1
EPOCH=100

ACT=relu
RATIO=1.0


for x in fmow,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.005
    do
        for MINR in 0.02
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
                --device $DEVICE
        done
    done
done