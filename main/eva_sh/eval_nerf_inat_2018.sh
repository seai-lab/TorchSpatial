#!/bin/bash
DIR=../models/nerf/ 
ENC=NeRF  

DATA=inat_2018
EVALDATA=val 
DEVICE=cuda:0

LR=0.01
LAYER=1 
HIDDIM=256 
FREQ=16 
MINR=0.1
MAXR=1.0
EPOCH=0

ACT=leakyrelu 
RATIO=1.0 


python3 main.py
    --save_results T\
    --load_super_model T\
    --spa_enc_type $ENC \
    --dataset $DATA \
    --eval_split $EVALDATA \
    --frequency_num $FREQ \
    --min_radius $MINR \
    --num_hidden_layer $LAYER \
    --hidden_dim $HIDDIM \
    --spa_f_act $ACT \
    --unsuper_lr 0.1 \
    --lr $LR \
    # --model_dir $DIR \
    --train_sample_ratio $RATIO \
    --device $DEVICE
