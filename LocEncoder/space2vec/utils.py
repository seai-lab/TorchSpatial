import numpy as np
import torch
import json
import os
import math
import pickle
from torch.utils.data.sampler import Sampler

from space2vec.SpatialRelationEncoder import *
from space2vec.module import *
# import models


def get_spa_enc_list():
    return ["gridcell", "gridcellnorm", "hexagridcell", "theory", "theorynorm", "theorydiag", "naive", "rbf"]


def get_ffn(params, input_dim, output_dim, f_act, context_str=""):

    return MultiLayerFeedForwardNN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_layers=params['num_hidden_layer'],
        dropout_rate=params['dropout'],
        hidden_dim=params['hidden_dim'],
        activation=f_act,
        use_layernormalize=params['use_layn'],
        skip_connection=params['skip_connection'],
        context_str=context_str)


def get_spa_encoder(train_locs, params, spa_enc_type, spa_embed_dim, extent, coord_dim=2,
                    frequency_num=16,
                    max_radius=10000, min_radius=1,
                    f_act="sigmoid", freq_init="geometric",
                    num_rbf_anchor_pts=100, rbf_kernal_size=10e2,
                    use_postmat=True,
                    device="cuda"):
    if spa_enc_type == "gridcell":
        ffn = get_ffn(params,
                      input_dim=int(4 * frequency_num),
                      output_dim=spa_embed_dim,
                      f_act=f_act,
                      context_str="GridCellSpatialRelationEncoder")
        spa_enc = GridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "gridcellnorm":
        ffn = get_ffn(params,
                      input_dim=int(4 * frequency_num),
                      output_dim=spa_embed_dim,
                      f_act=f_act,
                      context_str="GridCellNormSpatialRelationEncoder")
        spa_enc = GridCellNormSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "hexagridcell":
        spa_enc = HexagonGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            dropout=params['dropout'],
            f_act=f_act,
            device=device)
    elif spa_enc_type == "theory":
        ffn = get_ffn(params,
                      input_dim=int(6 * frequency_num),
                      output_dim=spa_embed_dim,
                      f_act=f_act,
                      context_str="TheoryGridCellSpatialRelationEncoder")
        spa_enc = TheoryGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "theorynorm":
        ffn = get_ffn(params,
                      input_dim=int(6 * frequency_num),
                      output_dim=spa_embed_dim,
                      f_act=f_act,
                      context_str="TheoryGridCellNormSpatialRelationEncoder")
        spa_enc = TheoryGridCellNormSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "theorydiag":
        spa_enc = TheoryDiagGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            dropout=params['dropout'],
            f_act=f_act,
            freq_init=freq_init,
            use_layn=params['use_layn'],
            use_post_mat=use_postmat,
            device=device)
    elif spa_enc_type == "naive":
        ffn = get_ffn(params,
                      input_dim=2,
                      output_dim=spa_embed_dim,
                      f_act=f_act,
                      context_str="NaiveSpatialRelationEncoder")
        spa_enc = NaiveSpatialRelationEncoder(
            spa_embed_dim,
            extent=extent,
            coord_dim=coord_dim,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "rbf":
        ffn = get_ffn(params,
                      input_dim=num_rbf_anchor_pts,
                      output_dim=spa_embed_dim,
                      f_act=f_act,
                      context_str="RBFSpatialRelationEncoder")
        spa_enc = RBFSpatialRelationEncoder(
            model_type="global",
            train_locs=train_locs,
            spa_embed_dim=spa_embed_dim,
            coord_dim=coord_dim,
            num_rbf_anchor_pts=num_rbf_anchor_pts,
            rbf_kernal_size=rbf_kernal_size,
            rbf_kernal_size_ratio=0,
            max_radius=max_radius,
            ffn=ffn,
            rbf_anchor_pt_ids=params['rbf_anchor_pt_ids'],
            device=device)
    elif spa_enc_type == "aodha":
        spa_enc = AodhaSpatialRelationEncoder(
            spa_embed_dim,
            extent=extent,
            coord_dim=coord_dim,
            num_hidden_layers=params['num_hidden_layer'],
            hidden_dim=params['hidden_dim'],
            use_post_mat=use_postmat,
            f_act=f_act)
    elif spa_enc_type == "none":
        assert spa_embed_dim == 0
        spa_enc = None
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc
