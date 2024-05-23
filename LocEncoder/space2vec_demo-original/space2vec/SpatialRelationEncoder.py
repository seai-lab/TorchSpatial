import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math

from space2vec.module import *
from space2vec.data_utils import *

"""
A Set of position encoder
"""



def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list

class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000, min_radius = 10,
            freq_init = "geometric",
            ffn=None, device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim 
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()


        self.input_embed_dim = self.cal_input_dim()

        
        self.ffn = ffn
        self.device = device


    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        # if self.freq_init == "random":
        #     # the frequence we use for each block, alpha in ICLR paper
        #     # self.freq_list shape: (frequency_num)
        #     self.freq_list = np.random.random(size=[self.frequency_num]) * self.max_radius
        # elif self.freq_init == "geometric":
        #     self.freq_list = []
        #     for cur_freq in range(self.frequency_num):
        #         base = 1.0/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))
        #         self.freq_list.append(base)

        #     self.freq_list = np.asarray(self.freq_list)
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)


    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis = 1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        # # convert to radius
        # coords_mat = coords_mat*math.pi/180
        
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        
        spr_embeds = self.make_input_embeds(coords)



        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        # return sprenc
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class GridCellNormSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000, min_radius = 10,
            freq_init = "geometric",
            ffn=None, device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellNormSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim 
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()


        self.input_embed_dim = self.cal_input_dim()

        
        self.ffn = ffn
        self.device = device


    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        # if self.freq_init == "random":
        #     # the frequence we use for each block, alpha in ICLR paper
        #     # self.freq_list shape: (frequency_num)
        #     self.freq_list = np.random.random(size=[self.frequency_num]) * self.max_radius
        # elif self.freq_init == "geometric":
        #     self.freq_list = []
        #     for cur_freq in range(self.frequency_num):
        #         base = 1.0/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))
        #         self.freq_list.append(base)

        #     self.freq_list = np.asarray(self.freq_list)
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)


    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis = 1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        # convert to radius
        coords_mat = coords_mat*math.pi/180
        
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        
        spr_embeds = self.make_input_embeds(coords)



        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        # return sprenc
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class HexagonGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000, dropout = 0.5, f_act = "sigmoid", device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(HexagonGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.spa_embed_dim = spa_embed_dim 

        self.input_embed_dim = self.cal_input_dim()

        self.post_linear = nn.Linear(self.input_embed_dim, self.spa_embed_dim)
        nn.init.xavier_uniform(self.post_linear.weight)
        self.dropout = nn.Dropout(p=dropout)

        # self.dropout_ = nn.Dropout(p=dropout)

        # self.post_mat = nn.Parameter(torch.FloatTensor(self.input_embed_dim, self.spa_embed_dim))
        # init.xavier_uniform_(self.post_mat)
        # self.register_parameter("spa_postmat", self.post_mat)

        self.f_act = get_activation_function(f_act, "HexagonGridCellSpatialRelationEncoder")
        
        self.device = device
    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq) + math.pi*2.0/3))
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq) + math.pi*4.0/3))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 3)


    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")


        # loop over all batches
        spr_embeds = []
        for cur_batch in coords:
            # loop over N context points
            cur_embeds = []
            for coords_tuple in cur_batch:
                cur_embeds.append(self.cal_coord_embed(coords_tuple))
            spr_embeds.append(cur_embeds)
        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        return sprenc


"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
"""
class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000,  min_radius = 1000, freq_init = "geometric", ffn = None, device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree


        self.input_embed_dim = self.cal_input_dim()


        # self.f_act = get_activation_function(f_act, "TheoryGridCellSpatialRelationEncoder")
        # self.dropout = nn.Dropout(p=dropout)

        # self.use_post_mat = use_post_mat
        # if self.use_post_mat:
        #     self.post_linear_1 = nn.Linear(self.input_embed_dim, 64)
        #     nn.init.xavier_uniform(self.post_linear_1.weight)
        #     self.post_linear_2 = nn.Linear(64, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear_2.weight)
        #     self.dropout_ = nn.Dropout(p=dropout)
        # else:
        #     self.post_linear = nn.Linear(self.input_embed_dim, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear.weight)
        self.ffn = ffn

        self.device = device
        
    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis = 1)


    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)


    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds
    
        
    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        # if self.use_post_mat:
        #     sprenc = self.post_linear_1(spr_embeds)
        #     sprenc = self.post_linear_2(self.dropout(sprenc))
        #     sprenc = self.f_act(self.dropout(sprenc))
        # else:
        #     sprenc = self.post_linear(spr_embeds)
        #     sprenc = self.f_act(self.dropout(sprenc))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds



"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
"""
class TheoryGridCellNormSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000,  min_radius = 1000, freq_init = "geometric", ffn = None, device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellNormSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree


        self.input_embed_dim = self.cal_input_dim()


        # self.f_act = get_activation_function(f_act, "TheoryGridCellSpatialRelationEncoder")
        # self.dropout = nn.Dropout(p=dropout)

        # self.use_post_mat = use_post_mat
        # if self.use_post_mat:
        #     self.post_linear_1 = nn.Linear(self.input_embed_dim, 64)
        #     nn.init.xavier_uniform(self.post_linear_1.weight)
        #     self.post_linear_2 = nn.Linear(64, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear_2.weight)
        #     self.dropout_ = nn.Dropout(p=dropout)
        # else:
        #     self.post_linear = nn.Linear(self.input_embed_dim, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear.weight)
        self.ffn = ffn

        self.device = device
        
    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis = 1)



    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)


    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # convert to radius
        coords_mat = coords_mat*math.pi/180

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds
    
        
    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        # if self.use_post_mat:
        #     sprenc = self.post_linear_1(spr_embeds)
        #     sprenc = self.post_linear_2(self.dropout(sprenc))
        #     sprenc = self.f_act(self.dropout(sprenc))
        # else:
        #     sprenc = self.post_linear(spr_embeds)
        #     sprenc = self.f_act(self.dropout(sprenc))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
We retrict the linear layer is block diagonal
"""
class TheoryDiagGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000, min_radius = 10, dropout = 0.5, f_act = "sigmoid", 
        freq_init = "geometric", use_layn=False, use_post_mat = False, device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryDiagGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        self.device = device

        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree


        self.input_embed_dim = self.cal_input_dim()

        assert self.spa_embed_dim % self.frequency_num == 0

        # self.post_linear = nn.Linear(self.frequency_num, 6, self.spa_embed_dim//self.frequency_num)
        
        # a block diagnal matrix
        self.post_mat = nn.Parameter(torch.FloatTensor(self.frequency_num, 6, self.spa_embed_dim//self.frequency_num).to(device))
        init.xavier_uniform_(self.post_mat)
        self.register_parameter("spa_postmat", self.post_mat)
        self.dropout = nn.Dropout(p=dropout)

        
        self.use_post_mat = use_post_mat
        if self.use_post_mat:
            self.post_linear = nn.Linear(self.spa_embed_dim, self.spa_embed_dim)
            self.dropout_ = nn.Dropout(p=dropout)
        

        self.f_act = get_activation_function(f_act, "TheoryDiagGridCellSpatialRelationEncoder")
        

    def cal_freq_list(self):
        # if self.freq_init == "random":
        #     # the frequence we use for each block, alpha in ICLR paper
        #     # self.freq_list shape: (frequency_num)
        #     self.freq_list = np.random.random(size=[self.frequency_num]) * self.max_radius
        # elif self.freq_init == "geometric":
        #     self.freq_list = []
        #     for cur_freq in range(self.frequency_num):
        #         base = 1.0/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))
        #         self.freq_list.append(base)

        #     self.freq_list = np.asarray(self.freq_list)
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)


    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis = 1)


    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        spr_embeds = angle_mat * self.freq_mat
        

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num, 6)
        spr_embeds[:, :, :, 0::2] = np.sin(spr_embeds[:, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, 1::2] = np.cos(spr_embeds[:, :, :, 1::2])  # dim 2i+1
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, frequency_num, 6)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, frequency_num, spa_embed_dim//frequency_num)
        sprenc = torch.einsum("bnfs,fsd->bnfd", (spr_embeds, self.post_mat))
        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        sprenc = sprenc.contiguous().view(batch_size, num_context_pt, self.spa_embed_dim)
        if self.use_post_mat:
            sprenc = self.dropout(sprenc)
            sprenc = self.f_act(self.dropout_(self.post_linear(sprenc)))
        else:
            # print(sprenc.size())
            sprenc = self.f_act(self.dropout(sprenc))

        return sprenc



class NaiveSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, extent, coord_dim = 2, ffn = None, device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            extent: (x_min, x_max, y_min, y_max)
        """
        super(NaiveSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim 
        self.coord_dim = coord_dim 
        self.extent = extent

        # self.post_linear = nn.Linear(self.coord_dim, self.spa_embed_dim)
        # nn.init.xavier_uniform(self.post_linear.weight)
        # self.dropout = nn.Dropout(p=dropout)

    

        # self.f_act = get_activation_function(f_act, "NaiveSpatialRelationEncoder")
        self.ffn = ffn

        self.device = device

        

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        coords_mat = coord_normalize(coords, self.extent)

        # spr_embeds: shape (batch_size, num_context_pt, coord_dim)
        spr_embeds = torch.FloatTensor(coords_mat).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

        # return sprenc





class RBFSpatialRelationEncoder(nn.Module):
    """
    Given a list of (X,Y), compute the distance from each pt to each RBF anchor points
    Feed into a MLP

    This is for global position encoding or relative/spatial context position encoding

    """
    def __init__(self, model_type, train_locs, spa_embed_dim, coord_dim = 2, 
                num_rbf_anchor_pts = 100, rbf_kernal_size = 10e2, rbf_kernal_size_ratio = 0.0, max_radius = 10000, ffn = None, 
                rbf_anchor_pt_ids = None,
                device = "cuda"): 
        """
        Args:
            train_locs: np.arrary, [batch_size, 2], location data
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            num_rbf_anchor_pts: the number of RBF anchor points
            rbf_kernal_size: the RBF kernal size
                        The sigma in https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf_kernal_size_ratio: if not None, (only applied on relative model)
                        different anchor pts have different kernal size :
                        dist(anchot_pt, origin) * rbf_kernal_size_ratio + rbf_kernal_size
            max_radius: the relative spatial context size in spatial context model
        """
        super(RBFSpatialRelationEncoder, self).__init__()
        self.model_type = model_type
        self.train_locs = train_locs
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim

        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_kernal_size = rbf_kernal_size
        self.rbf_kernal_size_ratio = rbf_kernal_size_ratio
        self.max_radius = max_radius
        self.rbf_anchor_pt_ids = rbf_anchor_pt_ids
        
        # calculate the coordinate matrix for each RBF anchor points
        self.cal_rbf_anchor_coord_mat()


        self.input_embed_dim = self.num_rbf_anchor_pts

        # self.use_layn = use_layn
        # self.use_post_mat = use_post_mat
        # if self.use_post_mat:
        #     self.post_linear1 = nn.Linear(self.input_embed_dim, 64)
        #     self.post_linear2 = nn.Linear(64, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear1.weight)
        #     nn.init.xavier_uniform(self.post_linear2.weight)
            
        # else:
        #     self.post_linear = nn.Linear(self.input_embed_dim, self.spa_embed_dim)
        #     nn.init.xavier_uniform(self.post_linear.weight)

        # self.dropout = nn.Dropout(p=dropout)

        # self.f_act = get_activation_function(f_act, "RBFSpatialRelationEncoder")

        self.ffn = ffn

        self.device = device
        
    def _random_sampling(self, item_tuple, num_sample):
        '''
        poi_type_tuple: (Type1, Type2,...TypeM)
        '''

        type_list = list(item_tuple)
        if len(type_list) > num_sample:
            return list(np.random.choice(type_list, num_sample, replace=False))
        elif len(type_list) == num_sample:
            return item_tuple
        else:
            return list(np.random.choice(type_list, num_sample, replace=True))

    def cal_rbf_anchor_coord_mat(self):
        if self.model_type == "global":
            assert self.rbf_kernal_size_ratio == 0
            # If we do RBF on location/global model, 
            # we need to random sample M RBF anchor points from training point dataset
            if self.rbf_anchor_pt_ids == None:
                self.rbf_anchor_pt_ids = self._random_sampling(np.arange(len(self.train_locs)), self.num_rbf_anchor_pts)


            # coords = []
            # for pid in rbf_anchor_pt_ids:
            #     coord = list(self.pointset.pt_dict[pid].coord)
            #     coords.append(coord)


            # self.rbf_coords: (num_rbf_anchor_pts, 2)
            # self.rbf_coords_mat = np.asarray(coords).astype(float)
            self.rbf_coords_mat = self.train_locs[self.rbf_anchor_pt_ids]

        elif self.model_type == "relative":
            # If we do RBF on spatial context/relative model,
            # We just random sample M-1 RBF anchor point in the relative spatial context defined by max_radius
            # The (0,0) is also an anchor point
            x_list = np.random.uniform(-self.max_radius, self.max_radius, self.num_rbf_anchor_pts)
            x_list[0] = 0.0
            y_list = np.random.uniform(-self.max_radius, self.max_radius, self.num_rbf_anchor_pts)
            y_list[0] = 0.0
            # self.rbf_coords: (num_rbf_anchor_pts, 2) 
            self.rbf_coords_mat = np.transpose(np.stack([x_list, y_list], axis=0))

            if self.rbf_kernal_size_ratio > 0:
                dist_mat = np.sqrt(np.sum(np.power(self.rbf_coords_mat, 2), axis = -1))
                # rbf_kernal_size_mat: (num_rbf_anchor_pts)
                self.rbf_kernal_size_mat = dist_mat * self.rbf_kernal_size_ratio + self.rbf_kernal_size

    def make_input_embeds(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, input_embed_dim)
        """
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for RBFSpatialRelationEncoder")

        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 1, 2)
        coords_mat = np.expand_dims(coords_mat, axis = 2)
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts, 2)
        coords_mat = np.repeat(coords_mat, self.num_rbf_anchor_pts, axis = 2)
        # compute (deltaX, deltaY) between each point and each RBF anchor points
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts, 2)
        coords_mat = coords_mat - self.rbf_coords_mat
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts=input_embed_dim)
        coords_mat = np.sum(np.power(coords_mat, 2), axis = 3)
        if self.rbf_kernal_size_ratio > 0: 
            spr_embeds = np.exp((-1*coords_mat)/(2.0 * np.power(self.rbf_kernal_size_mat, 2)))
        else:
            # spr_embeds: shape (batch_size, num_context_pt, num_rbf_anchor_pts=input_embed_dim)
            spr_embeds = np.exp((-1*coords_mat)/(2.0 * np.power(self.rbf_kernal_size, 2)))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)


        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
     
        # if self.use_post_mat:
        #     spr_embeds = self.dropout(self.post_linear1(spr_embeds))
        #     spr_embeds = self.post_linear2(spr_embeds)
        #     sprenc = self.f_act(spr_embeds)
        # else:
        #     sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


