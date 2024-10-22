#!/bin/bash

DIR=../models/wrap_ffn/

ENC=wrap_ffn

DATA=nabirds
META=ebird_meta
EVALDATA=test

DEVICE=cuda:2

LR=0.002
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.005
MAXR=1
EPOCH=30

ACT=relu
RATIO=1.0


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
