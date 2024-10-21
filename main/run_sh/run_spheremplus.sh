#!/bin/bash

DIR=../models_reg/sphere2vec_sphereMplus/

ENC=Sphere2Vec-sphereM+

DATA=mosaiks_nightlights
META=ebird_meta
EVALDATA=test

DEVICE=cuda:3

LR=0.0005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.001
MAXR=1
EPOCH=100


ACT=leakyrelu
RATIO=1.0


for LR in 0.0005 0.0009 0.0001 #0.00001 #0.00002 0.00005 #0.0005 #0.00005
do
    for FREQ in 64 32
    do
    for HIDDIM in 512 1024 256
    do
    python3 main.py \
        --save_results T\
        --load_super_model F\
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
        --embed_dim_before_regress 725\
        --unsuper_lr 0.1 \
        --lr $LR \
        --model_dir $DIR \
        --num_epochs $EPOCH \
        --train_sample_ratio $RATIO \
        --device $DEVICE
    done
    done
done

