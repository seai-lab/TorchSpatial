#!/bin/bash

DIR=../models/rff_test/

ENC=wrap_ffn

DATA=birdsnap
META=ebird_meta
EVALDATA=test

DEVICE=cuda:3
# model_birdsnap_ebird_meta_wrap_ffn_inception_v3_0.0010_64_0.1000000_1_512.pth.tar
LR=0.001
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.1
MAXR=1
EPOCH=0

ACT=relu
RATIO=1.0


python3 train_unsuper.py \
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
