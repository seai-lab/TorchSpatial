#!/bin/bash
DIR=../models/nerf/ 
ENC=NeRF  

# DATA=birdsnap 
DATA=inat_2017
# DATA=inat_2018
# DATA=nabirds
# META=ebird_meta 
# META=orig_meta
EVALDATA=test  
DEVICE=cuda:3

LR=0.005 
LAYER=1 
HIDDIM=512 
FREQ=64 
MINR=0.001 
# MAXR=1
# EPOCH=2

ACT=relu 
RATIO=1.0 

for x in inat_2017,ebird_meta,val  inat_2018,ebird_meta,val
do
    IFS=',' read DATA  META  EVALDATA <<< "${x}"
    for LR in 0.01 0.005 0.002 0.001 0.0005 0.00005
    do
        for LAYER in 1 2 3 4
        do
            for HIDDIM in 256 512 1024
            do
                for FREQ in 16 32 64
                do
                    for MINR in 0.10 0.05 0.02 0.01 0.005 0.001 0.0001
                    do
                        for ACT in relu leakyrelu sigmoid
                        do
                        python3 main.py
                            --spa_enc_type $ENC \
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
            done
        done
    done
done