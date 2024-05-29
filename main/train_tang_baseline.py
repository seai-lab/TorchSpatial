from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
from torchvision import datasets, transforms

from paths import get_paths
from scipy import sparse

import models
import datasets as dt
import utils as ut


class FeatureDataLoader(torch.utils.data.Dataset):
    def __init__(self, loc_feats, net_feats, labels, num_classes, is_train, params):
        self.loc_feats   = ut.convert_loc_to_tensor(loc_feats)
        self.loc_encoding = params['loc_encoding']
        self.net_feats   = torch.from_numpy(net_feats)
        self.labels      = torch.from_numpy(labels)
        self.is_train    = is_train
        self.num_classes = num_classes
        self.grid_size = params['grid_size']

    def __len__(self):
        return len(self.loc_feats)

    def __getitem__(self, index):
        op = {}
        if self.loc_encoding == 'discrete':
            op['loc_feat']  = torch.zeros(self.grid_size[0], self.grid_size[1])
            xx = int(((self.loc_feats[index, 0]+1)/2.0)*op['loc_feat'].shape[1])
            yy = int(((self.loc_feats[index, 1]+1)/2.0)*op['loc_feat'].shape[0])
            op['loc_feat'][yy, xx] = 1
            op['loc_feat'] = op['loc_feat'].reshape(op['loc_feat'].shape[0]*op['loc_feat'].shape[1])
        else:
            op['loc_feat'] = self.loc_feats[index, :]

        op['net_feat']  = self.net_feats[index, :]
        op['loc_class'] = self.labels[index]
        return op


def train(params, model, device, train_loader, optimizer, epoch, split_name):
    # adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    class_inv = torch.from_numpy(params['class_inv_freq']).to(device)
    model.train()
    print(split_name + ' epoch: {}'.format(epoch))
    for batch_idx, data in enumerate(train_loader):
        loc_feat = data['loc_feat'].to(device)
        net_feat = data['net_feat'].to(device)
        target   = data['loc_class'].to(device)

        optimizer.zero_grad()
        output = model(loc_feat, net_feat)
        #loss = F.cross_entropy(output, target, weight=class_inv)
        loss = F.nll_loss(output, target, weight=class_inv)
        loss.backward()
        optimizer.step()
        if (batch_idx % params['log_interval'] == 0) and (batch_idx != 0):
            print('[{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(loc_feat), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(params, model, device, test_loader, split_name, print_res=True, save_op=False):
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    with torch.no_grad():
        for data in test_loader:
            loc_feat = data['loc_feat'].to(device)
            net_feat = data['net_feat'].to(device)
            target   = data['loc_class'].to(device)
            output   = model(loc_feat, net_feat)
            if save_op:
                preds.append(torch.exp(output).data.cpu().numpy())
            #test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            test_loss = F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    if print_res:
        print('\n' + split_name + ' set: avg loss: {:.4f}, acc: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))

    if save_op:
        return test_acc, np.vstack(preds)
    else:
        return test_acc


def main():

    params = {}
    params['dataset'] = 'inat_2017'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    params['batch_size'] = 1024
    params['epochs'] = 30
    params['lr_decay'] = 0.98
    params['lr'] = 0.0001
    params['log_interval'] = 30
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params['eval_split'] = 'val'
    params['embedding_dim'] = 256
    params['grid_size'] = [180, 360]  # [100, 200], [36, 72]
    params['loc_encoding'] = 'gps'  # 'none', 'gps', 'discrete'
    params['meta_type'] = 'ebird_meta'
    params['inat2018_resolution'] = 'standard'
    if params['dataset'] in ['birdsnap', 'nabirds']:
        params['save_path'] = '../models/bl_tang_' + params['dataset'] + '_' + params['meta_type'] + '_' + params['loc_encoding'] + '.pth.tar'
    else:
        params['save_path'] = '../models/bl_tang_' + params['dataset'] + '_' + params['loc_encoding'] + '.pth.tar'
    params['use_loc'] = True
    if params['loc_encoding'] == 'none':
        params['use_loc'] = False


    print('\nDataset : ' + params['dataset'])
    print('Enc     : ' + params['loc_encoding'])
    print('Output  : ' + params['save_path'] + '\n')


    # load data and features
    op = dt.load_dataset(params, params['eval_split'], True, True, True, True, True)
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
    params['num_classes'] = op['num_classes']
    val_preds = op['val_preds']
    val_feats = op['val_feats']
    train_feats = op['train_feats']


    params['net_feats_dim'] = train_feats.shape[1]
    train_dataset = FeatureDataLoader(train_locs, train_feats, train_classes, params['num_classes'], True, params)
    train_loader  = torch.utils.data.DataLoader(train_dataset, num_workers=4, pin_memory=True,
                                                batch_size=params['batch_size'], shuffle=True)

    val_dataset = FeatureDataLoader(val_locs, val_feats, val_classes, params['num_classes'], False, params)
    val_loader  = torch.utils.data.DataLoader(val_dataset, num_workers=4, pin_memory=True,
                                              batch_size=params['batch_size'], shuffle=False)

    inputs_train = next(iter(train_loader))
    params['loc_feat_size'] = inputs_train['loc_feat'].shape[1]
    if params['use_loc'] == False:
        print('Not using location info')
    else:
        print('location feature size {}'.format(params['loc_feat_size']))

    model = models.TangNet(params['loc_feat_size'], params['net_feats_dim'],
                           params['embedding_dim'], params['num_classes'], params['use_loc']).to(params['device'])

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    #optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

    # weight by inverse class count to handle imbalance
    class_un, class_cnt_n = np.unique(train_classes, return_counts=True)
    class_cnt = np.ones(params['num_classes'], dtype=np.float32)
    class_cnt[class_un] += class_cnt_n
    params['class_inv_freq'] = float(class_cnt.sum()) / (params['num_classes'] * class_cnt.astype(np.float32))

    best_acc = 0.0
    best_epoch = 1
    for epoch in range(1, params['epochs'] + 1):
        train(params, model, params['device'], train_loader, optimizer, epoch, 'train')
        val_acc = test(params, model, params['device'], val_loader, 'val', True, False)

        if val_acc > best_acc:
            print('* Saving new best model to : ' + params['save_path'])
            best_acc = val_acc
            best_epoch = epoch
            op_state = {'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'params' : params}
            torch.save(op_state, params['save_path'])


if __name__== "__main__":
    main()
