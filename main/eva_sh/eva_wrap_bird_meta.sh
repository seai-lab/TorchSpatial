#!/bin/bash

DIR=../models/wrap/

ENC=wrap

# DATA=birdsnap
# DATA=inat_2017
# DATA=inat_2018
DATA=birdsnap
META=ebird_meta
#META=orig_meta
EVALDATA=test

DEVICE=cuda:3

LR=0.005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.0001
MAXR=1
EPOCH=30

ACT=relu
RATIO=1.0


for x in birdsnap,ebird_meta,test  #nabirds,ebird_meta,test   #inat_2018,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.005
    do
        for MINR in 0.0001
        do
            python3 train_unsuper.py \
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