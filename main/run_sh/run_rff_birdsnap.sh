#!/bin/bash

DIR=../models/rff/

ENC=rff

# DATA=birdsnap
# DATA=inat_2017
# DATA=inat_2018
DATA=nabirds
# META=ebird_meta
META=orig_meta
EVALDATA=test

DEVICE=cuda:2

LR=0.005
LAYER=1
HIDDIM=512
FREQ=64
MINR=0.000001
# MAXR=1
EPOCH=30

ACT=relu
RATIO=1.0

#num_rbf_anchor_pts = 200 #[100, 200, 500]
#rbf_kernel_size = # [0.5, 1, 2, 10]
KERNELSIZE=1
# the first round is 0.5

for x in birdsnap,orig_meta,test  birdsnap,ebird_meta,test  nabirds,ebird_meta,test  #inat_2018,ebird_meta,val  inat_2018,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.01 0.005 0.002 0.001 0.0005
    do
        for MINR in 0.10 0.05 0.02 0.01 0.005 0.001 0.0001
        do
            for KERNELSIZE in 0.5 1 2
            do
                python3 main.py \
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
                    --num_epochs $EPOCH \
                    --train_sample_ratio $RATIO \
                    --device $DEVICE \
                    --rbf_kernel_size $KERNELSIZE
            done
        done
    done
done