#!/bin/bash

DIR=../models/sphere2vec_sphereMplus/

ENC=Sphere2Vec-sphereM+

# DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
DATA=birdsnap
META=orig_meta
#META=ebird_meta
EVALDATA=test

DEVICE=cuda:1

LR=0.001
LAYER=1
HIDDIM=512
FREQ=32
MINR=0.000001
MAXR=1
EPOCH=30


ACT=leakyrelu
RATIO=1.0


for x in birdsnap,orig_meta,test  #nabirds,ebird_meta,test  #birdsnap,ebird_meta,test     #inat_2018,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.001 #0.002 0.005 0.0005
    do
        for MINR in 0.000001 #0.0001
        do
            for FREQ in 32
            do
                python3 main.py \
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
            done
        done
    done
done
