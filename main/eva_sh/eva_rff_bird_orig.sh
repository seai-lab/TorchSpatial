#!/bin/bash

DIR=../models/rff_test/

ENC=rff

DATA=birdsnap
META=orig_meta
EVALDATA=test

DEVICE=cuda:3
# model_birdsnap_orig_meta_rff_inception_v3_0.0020_64_0.1000000_1_512_1.0.pth
LR=0.002
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.1
MAXR=1
KERNELSIZE=1
################# Please set “--num_epochs” to be 0, because you do not want further train the model. #################
EPOCH=30

ACT=relu
RATIO=1.0

################# Now you have a set of hyperparameter fixed, so cancel the loops #################
################# Please set “–save_results” to be T AND “--load_super_model” to be T #################


python3 train_unsuper.py \
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
    --unsuper_lr 0.1 \
    --lr $LR \
    --model_dir $DIR \
    --num_epochs $EPOCH \
    --train_sample_ratio $RATIO \
    --device $DEVICE \
    --rbf_kernel_size $KERNELSIZE

