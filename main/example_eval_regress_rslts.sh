#!/bin/bash

DIR=../models/nemin_for_test/

ENC=Space2Vec-grid #Space2Vec-grid #rff #wrap sustainbench_asset_index

DATA=mosaiks_nightlights # mosaiks_forest_cover mosaiks_elevation mosaiks_nightlights # mosaiks_population #sustainbench_asset_index # nabirds

# DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
META=ebird_meta
# META=orig_meta
EVALDATA=test

DEVICE=cuda:3

LAYER=2
HIDDIM=1024
FREQ=64
MINR=0.05
MAXR=360

EPOCH=40

ACT=leakyrelu

python3 train_unsuper.py \
    --save_results F\
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
    --embed_dim_before_regress 218\
    --lr 0.0009 \
    --model_dir $DIR \
    --num_epochs $EPOCH \
    --device $DEVICE

