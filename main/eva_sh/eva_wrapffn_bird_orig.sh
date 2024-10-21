#!/bin/bash

DIR=../models/wrap_ffn/

ENC=wrap_ffn

DATA=birdsnap
META=orig_meta
EVALDATA=test

DEVICE=cuda:3

LR=0.002
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.01
MAXR=1
EPOCH=0

ACT=leakyrelu
RATIO=1.0


python3 main.py \
    --save_results T\
    --load_super_model T\
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
