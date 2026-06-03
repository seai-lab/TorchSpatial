#!/bin/bash

DIR=../models/resiren/

ENC=ReSiren

# -----------------------------------------------------------------------------------------------
# ReSiren is a SIREN with residual connections. The residual connections exist specifically to
# let the network scale to MANY layers, which is where its advantage over a plain SIREN shows up
# (Climplicit, arXiv:2504.05089). It therefore must be benchmarked with a DEPTH sweep -- the repo
# default of --num_hidden_layer 1 would collapse it to a shallow SIREN and hide its advantage.
# In Climplicit the encoder used 16 SIREN layers, a 512 hidden dim, and a 256-d output.
#
# Note: ReSiren uses the Direct [-1,1] positional encoding and a fixed sine activation, so the
# positional-encoding knobs that the other encoders sweep (--frequency_num, --min_radius,
# --max_radius, --spa_f_act) have no effect here and are intentionally omitted. The only size
# knobs that matter are --num_hidden_layer (SIREN depth) and --hidden_dim (SIREN width); the
# output dim is the shared --num_filts (default 256).
# -----------------------------------------------------------------------------------------------

EVALDATA=val
DEVICE=cuda:0

LR=0.01
LAYER=16          # ReSiren scales with depth; the repo default (1) is far too shallow for it
HIDDIM=512
EPOCH=10
RATIO=1.0

for x in inat_2017,ebird_meta,val  inat_2018,ebird_meta,val  birdsnap,ebird_meta,val  nabirds,ebird_meta,val
do
    IFS=',' read DATA META EVALDATA <<< "${x}"
    for LR in 0.01 0.005 0.001
    do
        for HIDDIM in 256 512 1024
        do
            for LAYER in 1 2 4 8 12 16
            do
                python3 main.py \
                    --spa_enc_type $ENC \
                    --meta_type $META \
                    --dataset $DATA \
                    --eval_split $EVALDATA \
                    --num_hidden_layer $LAYER \
                    --hidden_dim $HIDDIM \
                    --unsuper_lr 0.1 \
                    --lr $LR \
                    --model_dir $DIR \
                    --num_epochs $EPOCH \
                    --train_sample_ratio $RATIO \
                    --device $DEVICE \
                    --save_result T
            done
        done
    done
done
