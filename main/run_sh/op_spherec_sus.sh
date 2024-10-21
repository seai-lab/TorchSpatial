#!/bin/bash

DIR=../models_reg/sphere2vec_sphereC/

ENC=Sphere2Vec-sphereC

DATA=sustainbench_asset_index
#sustainbench_asset_index
#sustainbench_under5_mort
#sustainbench_water_index
#sustainbench_women_bmi
#sustainbench_women_edu
#sustainbench_sanitation_index

META=ebird_meta
EVALDATA=test

DEVICE=cuda:2

LR=0.0005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.001
MAXR=1
EPOCH=60


ACT=leakyrelu
RATIO=1.0



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
    --embed_dim_before_regress 218\
    --unsuper_lr 0.1 \
    --lr $LR \
    --model_dir $DIR \
    --num_epochs $EPOCH \
    --train_sample_ratio $RATIO \
    --device $DEVICE
