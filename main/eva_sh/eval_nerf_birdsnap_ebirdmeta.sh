#!/bin/bash
DIR=../models/nerf/ 
ENC=NeRF  

DATA=birdsnap 
META=ebird_meta 
EVALDATA=test  
DEVICE=cuda:0 

LR=0.01
LAYER=1 
HIDDIM=256 
FREQ=64 
MINR=0.02 
MAXR=1
EPOCH=0

ACT=leakyrelu 
RATIO=1.0 

python3 train_unsuper.py \
    --save_results T\
    --load_super_model T\
    --num_epochs $EPOCH \
    --spa_enc_type $ENC \
    --meta_type $META\
    --dataset $DATA \
    --eval_split $EVALDATA \
    --frequency_num $FREQ \
    --min_radius $MINR \
    --max_radius $MAXR \
    --num_hidden_layer $LAYER \
    --hidden_dim $HIDDIM \
    --spa_f_act $ACT \
    --unsuper_lr 0.1 \
    --lr $LR \
    --model_dir $DIR \
    --train_sample_ratio $RATIO \
    --device $DEVICE
