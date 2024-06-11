#!/bin/bash

DIR=../models/spherical_harmonics/

ENC=spherical_harmonics

DATA=birdsnap
# DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
META=ebird_meta
# META=orig_meta
EVALDATA=test

DEVICE=cuda:0

LR=0.005
LAYER=1
L=8
HIDDIM=512
FREQ=32
MINR=0.0005
MAXR=1
EPOCH=10

ACT=relu
RATIO=1.0


for x in yfcc,ebird_meta,test
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.01
    do
        for MINR in 0.0001 
        do
            for HIDDIM in 512
            do
                for LAYER in 3
                do
                    python3 train_unsuper.py \
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
                        --device $DEVICE\
			--save_result T
                done
            done    
        done
    done
done
