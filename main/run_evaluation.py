"""
Script to evaluate different spatio-temporal priors.
"""

import numpy as np
import json
from scipy import sparse
import torch
import math
import pandas as pd
import os
from sklearn.neighbors import BallTree, DistanceMetric
from argparse import ArgumentParser

from paths import get_paths
import utils as ut
import datasets as dt
import baselines as bl
import models

def compute_acc_batch(val_preds, val_classes, val_split, val_feats=None, train_classes=None,
                train_feats=None, prior_type='no_prior', prior=None, hyper_params=None, batch_size = 1024, 
                logger = None):
    '''
    Computes accuracy on held out set with a specified prior. Not very efficient
    as it loops though each example one at a time.
    Args:
        val_preds: CNN pretrained model's image prediction of class [batch_size, num_classes]
        val_classes: [batch_size, 1], the list of image category id
        val_split: for bridsnap, np.ones() (batch_size)
        val_feats: the input location features, shape [batch_size, x]
        train_classes:
        train_feats:
        prior_type: 'geo_net'
        prior: the model itself
    Return:
        pred_classes: (batch_size), the list of joint predicted image category
    '''
    pred_list = []
    val_classes_list = []
    inds_list = []
    
    for start_ind in range(0, len(val_classes), batch_size):
        end_ind = min(start_ind + batch_size, len(val_classes))
        inds = np.asarray(list(range(start_ind, end_ind)))
        cnn_pred = val_preds[inds, :]

        # select the type of prior to be used
        if prior_type == 'no_prior':
            pred = cnn_pred

        elif prior_type == 'train_freq':
            # prior: (1, num_classes)
            pred = cnn_pred*prior

        elif prior_type == 'nn_dist':
            geo_prior = []
            for ind in range(start_ind, end_ind):
                geo_prior.append(bl.compute_neighbor_prior(train_classes, val_preds.shape[1],
                                val_feats[ind, :], prior, hyper_params, ptype='distance'))
            # geo_prior: (batch_size, num_classes)
            geo_prior = np.concatenate(geo_prior, axis = 0)
            pred = cnn_pred*geo_prior

        elif prior_type == 'nn_knn':
            geo_prior = []
            for ind in range(start_ind, end_ind):
                geo_prior.append(bl.compute_neighbor_prior(train_classes, val_preds.shape[1],
                               val_feats[ind, :], prior, hyper_params, ptype='knn'))
            # geo_prior: (batch_size, num_classes)
            geo_prior = np.concatenate(geo_prior, axis = 0)
            pred = cnn_pred*geo_prior

        elif prior_type == 'kde':
            geo_prior = []
            for ind in range(start_ind, end_ind):
                geo_prior.append(bl.kde_prior(train_classes, train_feats, val_preds.shape[1],
                               val_locs[ind, :], prior, hyper_params))
            # geo_prior: (batch_size, num_classes)
            geo_prior = np.concatenate(geo_prior, axis = 0)
            pred = cnn_pred*geo_prior

        elif prior_type == 'grid':
            geo_prior = prior.eval(val_feats[inds, :])
            pred = cnn_pred*geo_prior

        elif prior_type in ['geo_net'] + ut.get_spa_enc_list():
            # if there is no location info won't use prior
            # cnn_pred: the pretrained CNN image class prediction distribution
            # cnn_pred = val_preds[inds, :]
            with torch.no_grad():
                # if all image have location infor
                loc_isnan = torch.isnan(val_feats[inds, 0]).cpu().data.numpy().astype(int)
                inds = inds[np.where(loc_isnan == 0)]
                # if torch.sum(torch.isnan(val_feats[inds, 0])).item() == 0:
                #     print("Hi!")
                # net_prior: (batch_size, num_classes), the spa_enc model image class prediction distribution
                net_prior = prior(val_feats[inds, :])
                net_prior = net_prior.cpu().data.numpy().astype(np.float64)

                cnn_pred = val_preds[inds, :]
                #net_prior /= net_prior.sum()  # does not matter for argmax
                pred = cnn_pred*net_prior

                val_classes_list.append(val_classes[inds])

                inds_list.append(inds)
 

        elif prior_type == 'tang_et_al':
            # if there is no location info won't use prior
            # pred = val_preds[ind, :]
            with torch.no_grad():
                loc_isnan = torch.isnan(val_feats[inds, 0]).data.numpy().astype(int)
                inds = inds[np.where(loc_isnan == 0)]
                # if torch.sum(torch.isnan(val_feats[inds, 0])).item() == 0:
                # takes location and network features as input
                pred = prior(val_feats['val_locs'][inds, :],
                                  val_feats['val_feats'][inds, :])
                pred = pred.cpu().data.numpy().astype(np.float64)

        # pred_list: (num_batch, batch_size, num_classes)
        pred_list.append(pred)


    # preds: (num_sample, num_classes)
    preds = np.concatenate(pred_list, axis = 0)
    # logger.info(preds.shape)
    

    if prior_type in ['geo_net'] + ut.get_spa_enc_list():
        val_classes_ = np.concatenate(val_classes_list, axis = 0)
        logger.info(val_classes_.shape)
        # ranks: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
        ranks = get_label_rank(loc_pred = preds, loc_class = val_classes_)
        inds_list = np.concatenate(inds_list, axis = 0)
    else:
        logger.info(val_classes.shape)
        # ranks: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
        ranks = get_label_rank(loc_pred = preds, loc_class = val_classes)
        inds_list = None

    top_k_acc = {}
    for kk in [1, 3, 5, 10]:
        top_k_acc[kk] = (ranks<=kk).astype(int)

    # print final accuracy
    # some datasets have mutiple splits. These are represented by integers for each example in val_split
    for ii, split in enumerate(np.unique(val_split)):
        logger.info(' Split ID: {}'.format(ii))
        inds1 = np.where(val_split == split)[0]
        if inds_list is not None:
            inds2 = sorted(list(set(list(inds1)).intersection( set(list(inds_list)) )))
            idx_map = dict(zip(list(inds_list), list(range(len(inds_list)))))
            inds = [idx_map[idx] for idx in inds2]
        else:
            inds = inds1
        for kk in np.sort(list(top_k_acc.keys())):
            logger.info(' Top {}\tacc (%):   {}'.format(kk, round(top_k_acc[kk][inds].sum()*100/len(inds1), 2)))

    pred_classes = list(np.argmax(preds, axis = -1))
    return pred_classes


def get_label_rank(loc_pred, loc_class):
    '''
    Args:
        loc_pred: np matrix, [batch_size, num_classes], the prediction probability distribution of each sample over all classes
        loc_class: np matrix, [batch_size], the ground truth class
    '''
    loc_pred_ = loc_pred
    # loc_pred_idx: [batch_size, num_classes], the reverse rank (large->small) of all classes based on the probability
    loc_pred_idx = np.argsort(loc_pred_, axis = -1)[:, ::-1]

    # the rank for each class in each sample
    ranks_ = np.argsort(loc_pred_idx, axis = -1) + 1

    loc_class_ = loc_class

    nids = np.arange(loc_pred_.shape[0])

    # rank_list: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
    rank_list = ranks_[nids, loc_class_]

    # num_classes = loc_pred_.shape[1]

    # loc_class_ = loc_class.cpu().data.numpy()
    
    # # loc_class_: [batch_size, num_classes], the correct class label for each sample
    # loc_class_ = np.repeat(np.expand_dims(loc_class_, axis = 1), num_classes, axis = 1)

    # # rank_list: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
    # rank_list = np.argmax(loc_pred_idx == loc_class_, axis = 1) + 1
    return rank_list


def compute_acc(val_preds, val_classes, val_split, val_feats=None, train_classes=None,
                train_feats=None, prior_type='no_prior', prior=None, hyper_params=None, 
                logger = None):
    '''
    Computes accuracy on held out set with a specified prior. Not very efficient
    as it loops though each example one at a time.
    Args:
        val_preds: CNN pretrained model's image prediction of class
        val_classes: [batch_size, 1], the list of image category id
        val_split: for bridsnap, np.ones() (batch_size)
        val_feats: the inpit location features, shape [batch_size, x]
        train_classes:
        train_feats:
        prior_type: 'geo_net'
        prior: the model itself
    Return:
        pred_classes: (batch_size), the list of joint predicted image category
    '''
    

    top_k_acc = {}
    for kk in [1, 3, 5, 10]:
        top_k_acc[kk] = np.zeros(len(val_classes))
    max_class = np.max(list(top_k_acc.keys()))
    pred_classes = [] # the list of joint predicted image category

    for ind in range(len(val_classes)):

        # select the type of prior to be used
        if prior_type == 'no_prior':
            pred = val_preds[ind, :]

        elif prior_type == 'train_freq':
            pred = val_preds[ind, :]*prior

        elif prior_type == 'nn_dist':
            geo_prior = bl.compute_neighbor_prior(train_classes, val_preds.shape[1],
                        val_feats[ind, :], prior, hyper_params, ptype='distance')
            pred = val_preds[ind, :]*geo_prior

        elif prior_type == 'nn_knn':
            geo_prior = bl.compute_neighbor_prior(train_classes, val_preds.shape[1],
                           val_feats[ind, :], prior, hyper_params, ptype='knn')
            pred = val_preds[ind, :]*geo_prior

        elif prior_type == 'kde':
            geo_prior = bl.kde_prior(train_classes, train_feats, val_preds.shape[1],
                           val_locs[ind, :], prior, hyper_params)
            pred = val_preds[ind, :]*geo_prior

        elif prior_type == 'grid':
            geo_prior = prior.eval(val_feats[ind, :])
            pred = val_preds[ind, :]*geo_prior

        elif prior_type in ['geo_net'] + ut.get_spa_enc_list():
            # if there is no location info won't use prior
            # pred: the pretrained CNN image class prediction distribution
            pred = val_preds[ind, :]
            with torch.no_grad():
                # if all image have location infor
                if torch.isnan(val_feats[ind, 0]).item() == 0:
                    # net_prior: (1, num_classes), the spa_enc model image class prediction distribution
                    net_prior = prior(val_feats[ind, :].unsqueeze(0))
                    net_prior = net_prior.cpu().data.numpy()[0, :].astype(np.float64)
                    #net_prior /= net_prior.sum()  # does not matter for argmax
                    pred = pred*net_prior

        elif prior_type == 'tang_et_al':
            # if there is no location info won't use prior
            pred = val_preds[ind, :]
            with torch.no_grad():
                if torch.isnan(val_feats['val_locs'][ind, 0]).item() == 0:
                    # takes location and network features as input
                    pred = prior(val_feats['val_locs'][ind, :].unsqueeze(0),
                                      val_feats['val_feats'][ind, :].unsqueeze(0))
                    pred = pred.cpu().data.numpy()[0, :].astype(np.float64)


        # store accuracy of prediction
        pred_classes.append(np.argmax(pred))
        top_N = np.argsort(pred)[-max_class:]
        for kk in top_k_acc.keys():
            if val_classes[ind] in top_N[-kk:]:
                top_k_acc[kk][ind] = 1

    # print final accuracy
    # some datasets have mutiple splits. These are represented by integers for each example in val_split
    for ii, split in enumerate(np.unique(val_split)):
        logger.info(' Split ID: {}'.format(ii))
        inds = np.where(val_split == split)[0]
        for kk in np.sort(list(top_k_acc.keys())):
            logger.info(' Top {}\tacc (%):   {}'.format(kk, round(top_k_acc[kk][inds].mean()*100, 2)))

    return pred_classes


def get_cross_val_hyper_params(eval_params):

    hyper_params = {}
    if eval_params['dataset'] == 'inat_2018':
        hyper_params['num_neighbors'] = 1500
        hyper_params['dist_type'] = 'euclidean'  # euclidean, haversine
        hyper_params['dist_thresh'] = 2.0  # kms if haversine - divide by radius earth
        hyper_params['gp_size'] = [180, 60]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'  # for KDE
        hyper_params['kde_quant'] = 5.0  # for KDE
        hyper_params['kde_nb'] = 700  # for KDE

    elif eval_params['dataset'] == 'inat_2017':
        hyper_params['num_neighbors'] = 1450
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 5.0
        hyper_params['gp_size'] = [45, 30]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 5.0
        hyper_params['kde_nb'] = 700

    elif eval_params['dataset'] == 'birdsnap' and eval_params['meta_type'] == 'ebird_meta':
        hyper_params['num_neighbors'] = 700
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 5.0
        hyper_params['gp_size'] = [30, 30]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 500

    elif eval_params['dataset'] == 'birdsnap' and eval_params['meta_type'] == 'orig_meta':
        hyper_params['num_neighbors'] = 100
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 9.0
        hyper_params['gp_size'] = [225, 60]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 600

    elif eval_params['dataset'] == 'nabirds':
        hyper_params['num_neighbors'] = 500
        hyper_params['dist_type'] = 'euclidean'
        hyper_params['dist_thresh'] = 6.0
        hyper_params['gp_size'] = [45, 60]
        hyper_params['pseudo_count'] = 2
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 600

    elif eval_params['dataset'] == 'yfcc':
        hyper_params['num_neighbors'] = 75
        hyper_params['dist_type'] = 'haversine'
        hyper_params['dist_thresh'] = 2.0/6371.4
        hyper_params['gp_size'] = [540, 150]
        hyper_params['pseudo_count'] = 3
        hyper_params['kde_dist_type'] = 'euclidean'
        hyper_params['kde_quant'] = 0.001
        hyper_params['kde_nb'] = 300

    return hyper_params


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="inat_2018") # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    parser.add_argument("--meta_type", type=str, default="ebird_meta") # orig_meta, ebird_meta
    parser.add_argument("--eval_split", type=str, default="val") # train, val, test
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_dir", type=str, default="../models/")
    parser.add_argument("--num_epochs", type=int, default=30)

    parser.add_argument("--spa_enc_type", type=str, default="sphere")
    parser.add_argument("--lr", type=float, default=0.001,
        help='learning rate')
    parser.add_argument("--frequency_num", type=int, default=32,
        help='The number of frequency used in the space encoder')
    parser.add_argument("--max_radius", type=float, default=1.0,
        help='The maximum frequency in the space encoder')
    parser.add_argument("--min_radius", type=float, default=0.000001,
        help='The minimum frequency in the space encoder')
    parser.add_argument("--num_hidden_layer", type=int, default=1,
        help='The number of hidden layer in the space encoder')
    parser.add_argument("--hidden_dim", type=int, default=512,
        help='The hidden dimention in feedforward NN in the (global) space encoder')

    parser.add_argument("--num_rbf_anchor_pts", type=int, default=200,
        help='The number of RBF anchor points used in in the space encoder')
    parser.add_argument("--rbf_kernel_size", type=float, default=1.0,
        help='The RBF kernel size in the "rbf" space encoder')
#     parser.add_argument("--rand_sample_weight", type=float, default=1.0,
#         help='The weight of rand sample loss')

    args = parser.parse_args()


    params = {}
    eval_params = {}

    # eval_params['spa_enc'] = "spheregridmixscale"

    # params['lr'] = 0.001
    # params['frequency_num'] = 64
    # params["min_radius"] = 0.000001
    # params['num_hidden_layer'] = 1
    # params['hidden_dim'] = 512

    eval_params['spa_enc'] = args.spa_enc_type

#     params['rand_sample_weight'] = args.rand_sample_weight
    params['lr'] = args.lr
    params['frequency_num'] = args.frequency_num
    params["min_radius"] = args.min_radius
    params['num_hidden_layer'] = args.num_hidden_layer
    params['hidden_dim'] = args.hidden_dim

    params['num_rbf_anchor_pts'] = args.num_rbf_anchor_pts
    params['rbf_kernel_size'] = args.rbf_kernel_size

    params["model_dir"] = args.model_dir
    params["num_epochs"] = args.num_epochs
    params["max_radius"] = args.max_radius

    
    eval_params['dataset'] = args.dataset  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    eval_params['eval_split'] = args.eval_split  # train, val, test
    eval_params['inat2018_resolution'] = 'standard' # 'standard' or 'high_res' - only valid for inat_2018
    eval_params['meta_type'] = args.meta_type  # orig_meta, ebird_meta - only for nabirds, birdsnap
    eval_params['model_type'] = '' # '_full_final', '_no_date_final', '_no_photographer_final', '_no_encode_final'
    eval_params['trained_models_root'] = args.model_dir  # location where trained models are stored
    eval_params['save_op'] = False

    

    # specify which algorithms to evaluate. Ours is 'geo_net'.
    #eval_params['algs'] = ['no_prior', 'train_freq', 'geo_net', 'tang_et_al', 'grid', 'nn_knn', 'nn_dist', 'kde']
    eval_params['algs'] = ['no_prior', eval_params['spa_enc']]

    # if torch.cuda.is_available():
    #     device = torch.device("cuda:1")
    #     eval_params['device'] = device
    # else:
    #     eval_params['device'] = 'cpu'
    eval_params['device'] = args.device

    # path to trained models
    meta_str = ''
    if eval_params['dataset'] in ['birdsnap', 'nabirds']:
        meta_str = '_' + eval_params['meta_type']

    

    # param_args = "{lr:.4f}_{freq:d}_{min_radius:.7f}_{num_hidden_layer:d}_{hidden_dim:d}".format(
    #     lr = params['lr'],
    #     freq = params['frequency_num'],
    #     min_radius = params["min_radius"],
    #     num_hidden_layer = params['num_hidden_layer'],
    #     hidden_dim = params['hidden_dim']
    #     )
    # if args.spa_enc_type == "rff":
    #     param_args += "_{rbf_kernel_size:.1f}".format(
    #         rbf_kernel_size = params['rbf_kernel_size']
    #         )
    # if args.spa_enc_type == "rbf":
    #     param_args += "_{num_rbf_anchor_pts:d}_{rbf_kernel_size:.1f}".format(
    #         num_rbf_anchor_pts = params['num_rbf_anchor_pts'],
    #         rbf_kernel_size = params['rbf_kernel_size']
    #         )
    param_args = ut.make_model_file_param_args(params, spa_enc_type = args.spa_enc_type)

    nn_model_path = "{}/model_{}{}_{}_{}.pth.tar".format(
        eval_params['trained_models_root'],
        eval_params['dataset'],
        meta_str,
        eval_params['spa_enc'],
        param_args) 
    nn_model_path_tang = "{}/bl_tang_{}{}_gps.pth.tar".format(
        eval_params['trained_models_root'],
        eval_params['dataset'],
        meta_str
        )

    eval_params['log_file_name'] = nn_model_path.replace(".pth.tar", ".log")
    logger = ut.setup_logging(eval_params['log_file_name'], console = True, filemode='a')
    # eval_params['logger'] = logger

    # if eval_params['spa_enc'] == "rbf":
    #     rbf_anchor_pt_ids_file_name = "{}rbf_pts_{}{}_{}{}.pkl".format(
    #         eval_params['trained_models_root'],
    #         eval_params['dataset'],
    #         meta_str,
    #         eval_params['spa_enc'],
    #         eval_params['model_type'])
    #     rbf_anchor_pt_ids = ut.pickle_load(rbf_anchor_pt_ids_file_name)

    # nn_model_path_tang = eval_params['trained_models_root'] + 'bl_tang_'+eval_params['dataset']+meta_str+'_gps.pth.tar'

    logger.info('Dataset    \t' + eval_params['dataset'])
    logger.info('Meta Type    \t' + eval_params['meta_type'])
    logger.info('Eval split \t' + eval_params['eval_split'])
    logger.info('Model Path: \t' + nn_model_path)

    for key in params:
        logger.info("{}: {}".format(key, params[key]))

    # load data and features
    if 'tang_et_al' in eval_params['algs']:
        op = dt.load_dataset(eval_params, eval_params['eval_split'], True, False, True, True, False)
    else:
        op = dt.load_dataset(eval_params, eval_params['eval_split'], True, False, True, False, False)

    train_locs = op['train_locs']
    train_classes = op['train_classes']
    train_users = op['train_users']
    train_dates = op['train_dates']
    val_locs = op['val_locs']
    val_classes = op['val_classes']
    val_users = op['val_users']
    val_dates = op['val_dates']
    class_of_interest = op['class_of_interest']
    classes = op['classes']
    num_classes = op['num_classes']
    val_preds = op['val_preds']
    val_split = op['val_split']

    # these hyper parameters have been cross validated for the baseline methods
    hyper_params = get_cross_val_hyper_params(eval_params)


    #
    # no prior
    #
    if 'no_prior' in eval_params['algs']:
        logger.info('\nNo prior')
        # pred_no_prior = compute_acc(val_preds, val_classes, val_split, prior_type='no_prior')
        pred_no_prior = compute_acc_batch(val_preds, val_classes, val_split, prior_type='no_prior', 
                                        batch_size = 1024, logger = logger)


    #
    # overall training frequency prior
    #
    if 'train_freq' in eval_params['algs']:
        logger.info('\nTrain frequency prior')
        # weight the eval predictions by the overall frequency of each class at train time
        cls_id, cls_cnt = np.unique(train_classes, return_counts=True)
        train_prior = np.ones(num_classes)
        train_prior[cls_id] += cls_cnt
        train_prior /= train_prior.sum()
        compute_acc(val_preds, val_classes, val_split, prior_type='train_freq', prior=train_prior, 
                    logger = logger)


    #
    # neural network spatio-temporal prior
    #
    # if 'geo_net' in eval_params['algs']:
    #     print('\nNeural net prior')
    #     print(' Model :\t' + os.path.basename(nn_model_path))
    #     net_params = torch.load(nn_model_path)
    #     params = net_params['params']

    #     # construct features
    #     val_locs_scaled = ut.convert_loc_to_tensor(val_locs)
    #     val_dates_scaled = torch.from_numpy(val_dates.astype(np.float32)*2 - 1)
    #     val_feats_net = ut.encode_loc_time(val_locs_scaled, val_dates_scaled, concat_dim=1, params=params)

    #     model = models.FCNet(params['num_feats'], params['num_classes'], params['num_filts'], params['num_users'])
    #     model.load_state_dict(net_params['state_dict'])
    #     model.eval()
    #     pred_geo_net = compute_acc(val_preds, val_classes, val_split, val_feats=val_feats_net, prior_type='geo_net', prior=model)

    #
    # spatial encoder neural network spatio-temporal prior
    #
    spa_enc_algs = set(ut.get_spa_enc_list() + ['geo_net'])
    spa_enc_algs = set(eval_params['algs']).intersection(spa_enc_algs)
    if len(spa_enc_algs) == 1:
        spa_enc_type = list(spa_enc_algs)[0]
        
        logger.info('\n{}'.format(spa_enc_type))
        logger.info(' Model :\t' + os.path.basename(nn_model_path))

        net_params = torch.load(nn_model_path)
        params = net_params['params']

        # construct features
        # val_feats_net: shape [batch_size, 2], torch.tensor
        val_feats_net = ut.generate_model_input_feats(
                spa_enc_type = params['spa_enc_type'], 
                locs = val_locs, 
                dates = val_dates, 
                params = params,
                device = eval_params['device'])

        model = ut.get_loc_model(
            train_locs = train_locs,
            params = params, 
            spa_enc_type = params['spa_enc_type'], 
            num_inputs = params['num_feats'], 
            num_classes = params['num_classes'], 
            num_filts = params['num_filts'], 
            num_users = params['num_users'], 
            device = eval_params['device'])

        model.load_state_dict(net_params['state_dict'])
        model.eval()
        pred_geo_net = compute_acc(val_preds, val_classes, val_split, val_feats=val_feats_net, prior_type=spa_enc_type, 
                                prior=model, logger = logger)
        # pred_geo_net = compute_acc_batch(val_preds, val_classes, val_split, val_feats=val_feats_net, prior_type=spa_enc_type, prior=model, batch_size = params['batch_size'])
    #
    # Tang et al ICCV 2015, Improving Image Classification with Location Context
    #
    if 'tang_et_al' in eval_params['algs']:
        logger.info('\nTang et al. prior')
        logger.info('  using model :\t' + os.path.basename(nn_model_path_tang))
        net_params = torch.load(nn_model_path_tang)
        params = net_params['params']

        # construct features
        val_feats_tang = {}
        val_feats_tang['val_locs']  = ut.convert_loc_to_tensor(val_locs)
        val_feats_tang['val_feats'] = torch.from_numpy(op['val_feats'])
        assert params['loc_encoding'] == 'gps'

        model = models.TangNet(params['loc_feat_size'], params['net_feats_dim'],
                               params['embedding_dim'], params['num_classes'], params['use_loc'])
        model.load_state_dict(net_params['state_dict'])
        model.eval()
        compute_acc(val_preds, val_classes, val_split, val_feats=val_feats_tang, prior_type='tang_et_al', 
                    prior=model, logger = logger)
        del val_feats_tang  # save memory

    #
    # discretized grid prior
    #
    if 'grid' in eval_params['algs']:
        logger.info('\nDiscrete grid prior')
        gp = bl.GridPrior(train_locs, train_classes, num_classes, hyper_params)
        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs, prior_type='grid', prior=gp,
                    hyper_params=hyper_params, logger = logger)


    #
    # setup look up tree for NN lookup based methods
    #
    if ('nn_knn' in eval_params['algs']) or ('nn_dist' in eval_params['algs']):
        if hyper_params['dist_type'] == 'haversine':
            nn_tree = BallTree(np.deg2rad(train_locs)[:,::-1], metric='haversine')
            val_locs_n = np.deg2rad(val_locs)
        else:
            nn_tree = BallTree(train_locs[:,::-1], metric='euclidean')
            val_locs_n = val_locs


    #
    # nearest neighbor prior - based on KNN
    #
    if 'nn_knn' in eval_params['algs']:
        logger.info('\nNearest neighbor KNN prior')
        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs_n, train_classes=train_classes,
                    prior_type='nn_knn', prior=nn_tree, hyper_params=hyper_params, logger = logger)


    #
    # nearest neighbor prior - based on distance
    #
    if 'nn_dist' in eval_params['algs']:
        logger.info('\nNearest neighbor distance prior')
        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs_n, train_classes=train_classes,
                    prior_type='nn_dist', prior=nn_tree, hyper_params=hyper_params, logger = logger)


    #
    # kernel density estimate e.g. BirdSnap CVPR 2014
    #
    if 'kde' in eval_params['algs']:
        logger.info('\nKernel density estimate prior')
        kde_params = {}
        train_classes_kde, train_locs_kde, kde_params['counts'] = bl.create_kde_grid(train_classes, train_locs, hyper_params)
        if hyper_params['kde_dist_type'] == 'haversine':
            train_locs_kde = np.deg2rad(train_locs_kde)
            val_locs_kde = np.deg2rad(val_locs)
            kde_params['nn_tree_kde'] = BallTree(train_locs_kde[:, ::-1], metric='haversine')
        else:
            val_locs_kde = val_locs
            kde_params['nn_tree_kde'] = BallTree(train_locs_kde[:, ::-1], metric='euclidean')

        compute_acc(val_preds, val_classes, val_split, val_feats=val_locs_kde, train_classes=train_classes_kde,
                    train_feats=train_locs_kde, prior_type='kde', prior=kde_params, hyper_params=hyper_params, 
                    logger = logger)


    if eval_params['save_op']:
        np.savez('model_preds', val_classes=val_classes, pred_geo_net=pred_geo_net,
            pred_no_prior=pred_no_prior, dataset=eval_params['dataset'],
            split=eval_params['eval_split'], model_type=eval_params['model_type'])
