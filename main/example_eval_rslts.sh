#!/bin/bash

DIR=../models/sphere2vec_sphereC/

ENC=Sphere2Vec-sphereC

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
HIDDIM=512
FREQ=64
MINR=0.001
MAXR=1
################# Please set “--num_epochs” to be 0, because you do not want further train the model. #################
EPOCH=0

ACT=relu
RATIO=1.0


################# Now you have a set of hyperparameter fixed, so cancel the loops #################
################# Please set “–save_results” to be T AND “--load_super_model” to be T #################
for x in birdsnap,ebird_meta,test   #inat_2017,ebird_meta,val   inat_2018,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.01 #0.005 0.002 0.001 0.0005
    do
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
       
    done
done