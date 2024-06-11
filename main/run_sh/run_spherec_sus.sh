#!/bin/bash

DIR=../models_reg/sphere2vec_sphereC/

ENC=Sphere2Vec-sphereC


# META=orig_meta
META=ebird_meta
EVALDATA=test

DEVICE=cuda:1

LR=0.0005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.001
MAXR=1
EPOCH=70


ACT=leakyrelu
RATIO=1.0


for x in sustainbench_asset_index  sustainbench_women_edu  sustainbench_sanitation_index
do
    IFS=',' read DATA <<< "${x}"
    for LR in 0.001 0.002 0.005 0.0005 #0.00005
    do
        for MINR in 0.10 0.001 0.0001 0.000001
        do
        python3 train_unsuper.py \
            --spa_enc_type $ENC \
            --meta_type $META\
            --dataset $DATA \
            --eval_split $EVALDATA \
            --frequency_num $FREQ \
            --min_radius $MINR \
            --num_hidden_layer $LAYER \
            --hidden_dim $HIDDIM \
            --spa_f_act $ACT \
            --unsuper_lr 0.1 \
            --lr $LR \
            --model_dir $DIR \
            --train_sample_ratio $RATIO \
            --device $DEVICE
        done
    done
done
