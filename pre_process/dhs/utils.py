import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint_rbf.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_checkpoint_rbf.pth.tar')
        torch.save(state, best_filepath)

def load_checkpoint(model, optimizer, checkpoint_dir, filename='best_checkpoint_rbf.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{filepath}' (epoch {start_epoch})")
        return model, optimizer, best_val_loss, start_epoch
    else:
        print(f"No checkpoint found at '{filepath}'")
        return model, optimizer, best_val_loss, 0


class RBFFeaturePositionEncoder(nn.Module):
    """
    Given a list of values, compute the distance from each point to each RBF anchor point.
    Feed into an MLP.
    This is for global position encoding or relative/spatial context position encoding.
    """

    def __init__(
        self,
        train_locs,
        coord_dim=1,
        num_rbf_anchor_pts=100,
        rbf_kernel_size=10e2,
        rbf_kernel_size_ratio=0.0,
        model_type="global",
        max_radius=10000,
        rbf_anchor_pt_ids=None,
        device="cuda",
    ):
        """
        Args:
            train_locs: np.array, [batch_size], location data
            num_rbf_anchor_pts: the number of RBF anchor points
            rbf_kernel_size: the RBF kernel size
            rbf_kernel_size_ratio: if not None, different anchor points have different kernel size
            max_radius: the relative spatial context size in spatial context model
        """
        super(RBFFeaturePositionEncoder, self).__init__()
        self.coord_dim = coord_dim
        self.model_type = model_type
        self.train_locs = train_locs.values if isinstance(train_locs, pd.Series) else train_locs
        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_kernel_size = rbf_kernel_size
        self.rbf_kernel_size_ratio = rbf_kernel_size_ratio
        self.max_radius = max_radius
        self.rbf_anchor_pt_ids = rbf_anchor_pt_ids
        self.device = device

        # Calculate the coordinate matrix for each RBF anchor point
        self.cal_rbf_anchor_coord_mat()

        self.pos_enc_output_dim = self.num_rbf_anchor_pts
        # print(f"Position encoding output dimension: {self.pos_enc_output_dim}")

    def _random_sampling(self, item_tuple, num_sample):
        """
        Randomly sample a given number of items.
        """
        type_list = list(item_tuple)
        if len(type_list) > num_sample:
            return list(np.random.choice(type_list, num_sample, replace=False))
        elif len(type_list) == num_sample:
            return item_tuple
        else:
            return list(np.random.choice(type_list, num_sample, replace=True))

    def cal_rbf_anchor_coord_mat(self):
        if self.model_type == "global":
            assert self.rbf_kernel_size_ratio == 0
            # If we do RBF on location/global model,
            # we need to random sample M RBF anchor points from training point dataset
            if self.rbf_anchor_pt_ids == None:
                self.rbf_anchor_pt_ids = self._random_sampling(
                    np.arange(len(self.train_locs)), self.num_rbf_anchor_pts
                )

            self.rbf_coords_mat = self.train_locs[self.rbf_anchor_pt_ids]

        elif self.model_type == "relative":
            # If we do RBF on spatial context/relative model,
            # We just ra ndom sample M-1 RBF anchor point in the relative spatial context defined by max_radius
            # The (0,0) is also an anchor point
            x_list = np.random.uniform(
                -self.max_radius, self.max_radius, self.num_rbf_anchor_pts
            )
            x_list[0] = 0.0
            y_list = np.random.uniform(
                -self.max_radius, self.max_radius, self.num_rbf_anchor_pts
            )
            y_list[0] = 0.0
            # self.rbf_coords: (num_rbf_anchor_pts, 2)
            self.rbf_coords_mat = np.transpose(np.stack([x_list, y_list], axis=0))

            if self.rbf_kernel_size_ratio > 0:
                dist_mat = np.sqrt(np.sum(np.power(self.rbf_coords_mat, 2), axis=-1))
                # rbf_kernel_size_mat: (num_rbf_anchor_pts)
                self.rbf_kernel_size_mat = (
                    dist_mat * self.rbf_kernel_size_ratio + self.rbf_kernel_size
                )

    def make_output_embeds(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, pos_enc_output_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            #print("coords",coords.shape)
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            print("coords type",type(coords))
            raise Exception("Unknown coords data type for RBFSpatialRelationEncoder")

        coords_mat = np.asarray(coords).astype(float)
        #print("coords_mat1",coords_mat.shape)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        
        coords_mat = np.repeat(coords_mat, self.num_rbf_anchor_pts, axis=1)
        #print("coords_mat2",coords_mat.shape)
        coords_mat = coords_mat - self.rbf_coords_mat.T
        #print("coords_mat3",coords_mat.shape)
        coords_mat = np.sum(np.power(coords_mat, 2), axis=-1)
        #print("coords_mat4",coords_mat.shape)

        if self.rbf_kernel_size_ratio > 0:
            spr_embeds = np.exp(
                (-1 * coords_mat) / (2.0 * np.power(self.rbf_kernel_size_mat, 2))
            )
        else:
            spr_embeds = np.exp(
                (-1 * coords_mat) / (2.0 * np.power(self.rbf_kernel_size, 2))
            )
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coordinates, compute their spatial relation embedding.
        Args:
            coords: a list or array with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            spr_embeds: Tensor with shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_output_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)
        return spr_embeds
