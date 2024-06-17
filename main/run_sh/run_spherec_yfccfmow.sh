#!/bin/bash

DIR=../models/sphere2vec_sphereC/

ENC=Sphere2Vec-sphereC

# DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
DATA=fmow
# META=orig_meta
META=ebird_meta
EVALDATA=val

DEVICE=cuda:3

LR=0.01
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.001
MAXR=1
EPOCH=100


ACT=leakyrelu
RATIO=1.0


for x in fmow,ebird_meta,val  #yfcc,ebird_meta,test  
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.01 #0.0005 0.001 0.002 0.005
    do
        for MINR in 0.001 #0.0005 0.0001
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
