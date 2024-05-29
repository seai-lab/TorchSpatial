#!/bin/bash

DIR=../models/sphere2vec_sphereC/

ENC=rbf

DATA=birdsnap
# DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
#META=ebird_meta
META=orig_meta
EVALDATA=test

DEVICE=cuda:0

LR=0.005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.001
MAXR=1
EPOCH=30

ACT=relu
RATIO=1.0


python3 train_unsuper.py \
            --save_results T\
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
