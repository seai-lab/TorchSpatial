#!/bin/bash

DIR=../models/tile/

ENC=tile_ffn

DATA=birdsnap
META=orig_meta
EVALDATA=test

DEVICE=cuda:3
#model_birdsnap_orig_meta_tile_ffn_inception_v3_0.00018903_32_0.0000019_1_512_leakyrelu.pth
LR=0.00018903
LAYER=1
HIDDIM=512
FREQ=32
MINR=0.0000019
MAXR=1
# KERNELSIZE=2
################# Please set “--num_epochs” to be 0, because you do not want further train the model. #################
EPOCH=0

ACT=leakyrelu
RATIO=1.0

################# Now you have a set of hyperparameter fixed, so cancel the loops #################
################# Please set “–save_results” to be T AND “--load_super_model” to be T #################


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

