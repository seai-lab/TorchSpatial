import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math

from module import *
from data_utils import *

from spherical_harmonics_ylm_numpy import get_positional_encoding

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

        log_timescale_increment = math.log(float(max_radius) / float(min_radius)) / (
            frequency_num * 1.0 - 1
        )

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment
        )

        freq_list = 1.0 / timescales
    elif freq_init == "nerf":
        """
        compute according to NeRF position encoding, 
        Equation 4 in https://arxiv.org/pdf/2003.08934.pdf 
        2^{0}*pi, ..., 2^{L-1}*pi
        """
        #

        freq_list = np.pi * np.exp2(np.arange(frequency_num).astype(float))

    return freq_list


class GridCellSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of(deltaX, deltaY), encode them using the position encoding function
    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies / wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_pos_enc_output_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_pos_enc_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:  # noqa: E721
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=pos_enc_output_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords(deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape(batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape(batch_size, num_context_pt, position_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)

        return spr_embeds


class GridCellSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim=64,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="GridCellSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = GridCellSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            # input_dim=int(4 * frequency_num),
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class GridCellNormSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        ffn=None,
        device="cuda",
    ):
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

        self.pos_enc_output_dim = self.cal_input_dim()

        self.ffn = ffn
        self.device = device

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
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
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        # convert to radius
        coords_mat = coords_mat * math.pi / 180

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=pos_enc_output_dim)
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

        spr_embeds = self.make_output_embeds(coords)

        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
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

    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        dropout=0.5,
        f_act="sigmoid",
        device="cuda",
    ):
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

        self.pos_enc_output_dim = self.cal_input_dim()

        self.post_linear = nn.Linear(self.pos_enc_output_dim, self.spa_embed_dim)
        nn.init.xavier_uniform(self.post_linear.weight)
        self.dropout = nn.Dropout(p=dropout)

        # self.dropout_ = nn.Dropout(p=dropout)

        # self.post_mat = nn.Parameter(torch.FloatTensor(self.pos_enc_output_dim, self.spa_embed_dim))
        # init.xavier_uniform_(self.post_mat)
        # self.register_parameter("spa_postmat", self.post_mat)

        self.f_act = get_activation_function(
            f_act, "HexagonGridCellSpatialRelationEncoder"
        )

        self.device = device

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(
                    math.sin(
                        self.cal_elementwise_angle(coord, cur_freq) + math.pi * 2.0 / 3
                    )
                )
                embed.append(
                    math.sin(
                        self.cal_elementwise_angle(coord, cur_freq) + math.pi * 4.0 / 3
                    )
                )
        # embed: shape (pos_enc_output_dim)
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
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # loop over all batches
        spr_embeds = []
        for cur_batch in coords:
            # loop over N context points
            cur_embeds = []
            for coords_tuple in cur_batch:
                cur_embeds.append(self.cal_coord_embed(coords_tuple))
            spr_embeds.append(cur_embeds)
        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
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


class TheoryGridCellSpatialRelationPositionEncoder(
    GridCellSpatialRelationPositionEncoder
):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=1000,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])  # 0
        self.unit_vec2 = np.asarray([-1.0 / 2.0, math.sqrt(3) / 2.0])  # 120 degree
        self.unit_vec3 = np.asarray([-1.0 / 2.0, -math.sqrt(3) / 2.0])  # 240 degree

        self.pos_enc_output_dim = self.cal_pos_enc_output_dim()

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis=1)

    def cal_pos_enc_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis=-1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate(
            [angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3],
            axis=-1,
        )
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=pos_enc_output_dim)
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
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds


class TheoryGridCellSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="TheoryGridCellSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = TheoryGridCellSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


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

    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        dropout=0.5,
        f_act="sigmoid",
        freq_init="geometric",
        use_layn=False,
        use_post_mat=False,
        device="cuda",
    ):
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
        self.unit_vec1 = np.asarray([1.0, 0.0])  # 0
        self.unit_vec2 = np.asarray([-1.0 / 2.0, math.sqrt(3) / 2.0])  # 120 degree
        self.unit_vec3 = np.asarray([-1.0 / 2.0, -math.sqrt(3) / 2.0])  # 240 degree

        self.pos_enc_output_dim = self.cal_input_dim()

        assert self.spa_embed_dim % self.frequency_num == 0

        # self.post_linear = nn.Linear(self.frequency_num, 6, self.spa_embed_dim//self.frequency_num)

        # a block diagnal matrix
        self.post_mat = nn.Parameter(
            torch.FloatTensor(
                self.frequency_num, 6, self.spa_embed_dim // self.frequency_num
            ).to(device)
        )
        init.xavier_uniform_(self.post_mat)
        self.register_parameter("spa_postmat", self.post_mat)
        self.dropout = nn.Dropout(p=dropout)

        self.use_post_mat = use_post_mat
        if self.use_post_mat:
            self.post_linear = nn.Linear(self.spa_embed_dim, self.spa_embed_dim)
            self.dropout_ = nn.Dropout(p=dropout)

        self.f_act = get_activation_function(
            f_act, "TheoryDiagGridCellSpatialRelationEncoder"
        )

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
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis=1)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis=-1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate(
            [angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3],
            axis=-1,
        )
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis=-2)
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

        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, frequency_num, 6)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, frequency_num, spa_embed_dim//frequency_num)
        sprenc = torch.einsum("bnfs,fsd->bnfd", (spr_embeds, self.post_mat))
        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        sprenc = sprenc.contiguous().view(
            batch_size, num_context_pt, self.spa_embed_dim
        )
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

    def __init__(self, spa_embed_dim, extent, coord_dim=2, ffn=None, device="cuda"):
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
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

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


class XYZSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (lon,lat), convert them to (x,y,z), and then encode them using the MLP

    """

    def __init__(self, coord_dim=2, device="cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            extent: (x_min, x_max, y_min, y_max)
        """
        super().__init__(coord_dim=coord_dim, device=device)

        self.pos_enc_output_dim = 3  # self.cal_pos_enc_output_dim()

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # lon: (batch_size, num_context_pt, 1), convert from degree to radius
        lon = np.deg2rad(coords_mat[:, :, :1])
        # lat: (batch_size, num_context_pt, 1), convert from degree to radius
        lat = np.deg2rad(coords_mat[:, :, 1:])

        # convert (lon, lat) to (x,y,z), assume in unit sphere with r = 1
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        # spr_embeds: (batch_size, num_context_pt, 3)
        spr_embeds = np.concatenate((x, y, z), axis=-1)
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

        # spr_embeds: (batch_size, num_context_pt, 3)
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = 3)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds


class XYZSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="XYZSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection
        self.ffn_context_str = ffn_context_str

        self.position_encoder = XYZSpatialRelationPositionEncoder(
            coord_dim=coord_dim, device=device
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class NERFSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (lon,lat), convert them to (x,y,z), and then encode them using the MLP

    """

    def __init__(self, coord_dim=2, frequency_num=16, freq_init="nerf", device="cuda"):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init

        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = 6 * frequency_num

    def cal_freq_list(self):
        """
        compute according to NeRF position encoding,
        Equation 4 in https://arxiv.org/pdf/2003.08934.pdf
        2^{0}*pi, ..., 2^{L-1}*pi
        """
        # freq_list: shape (frequency_num)
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, None, None)

    def cal_freq_mat(self):
        # freq_mat shape: (1, frequency_num)
        freq_mat = np.expand_dims(self.freq_list, axis=0)
        # self.freq_mat shape: (3, frequency_num)
        self.freq_mat = np.repeat(freq_mat, 3, axis=0)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # lon: (batch_size, num_context_pt, 1), convert from degree to radius
        lon = np.deg2rad(coords_mat[:, :, :1])
        # lat: (batch_size, num_context_pt, 1), convert from degree to radius
        lat = np.deg2rad(coords_mat[:, :, 1:])

        # convert (lon, lat) to (x,y,z), assume in unit sphere with r = 1
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        # coords_mat: (batch_size, num_context_pt, 3)
        coords_mat = np.concatenate((x, y, z), axis=-1)
        # coords_mat: (batch_size, num_context_pt, 3, 1)
        coords_mat = np.expand_dims(coords_mat, axis=-1)
        # coords_mat: (batch_size, num_context_pt, 3, frequency_num)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=-1)
        # coords_mat: (batch_size, num_context_pt, 3, frequency_num)
        coords_mat = coords_mat * self.freq_mat

        # coords_mat: (batch_size, num_context_pt, 6, frequency_num)
        spr_embeds = np.concatenate([np.sin(coords_mat), np.cos(coords_mat)], axis=2)

        # spr_embeds: (batch_size, num_context_pt, 6*frequency_num)
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
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # spr_embeds: (batch_size, num_context_pt, 3)
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = 6*frequency_num)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds


class NERFSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        device="cuda",
        frequency_num=16,
        freq_init="nerf",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="NERFSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = NERFSpatialRelationPositionEncoder(
            coord_dim=coord_dim, frequency_num=frequency_num, device=device
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class RBFSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (X,Y), compute the distance from each pt to each RBF anchor points
    Feed into a MLP

    This is for global position encoding or relative/spatial context position encoding
    """

    def __init__(
        self,
        model_type,
        train_locs,
        coord_dim=2,
        num_rbf_anchor_pts=100,
        rbf_kernel_size=10e2,
        rbf_kernel_size_ratio=0.0,
        max_radius=10000,
        rbf_anchor_pt_ids=None,
        device="cuda",
    ):
        """
        Args:
            train_locs: np.arrary, [batch_size, 2], location data
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            num_rbf_anchor_pts: the number of RBF anchor points
            rbf_kernel_size: the RBF kernel size
                        The sigma in https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf_kernel_size_ratio: if not None, (only applied on relative model)
                        different anchor pts have different kernel size :
                        dist(anchot_pt, origin) * rbf_kernel_size_ratio + rbf_kernel_size
            max_radius: the relative spatial context size in spatial context model
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.model_type = model_type
        self.train_locs = train_locs
        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_kernel_size = rbf_kernel_size
        self.rbf_kernel_size_ratio = rbf_kernel_size_ratio
        self.max_radius = max_radius
        self.rbf_anchor_pt_ids = rbf_anchor_pt_ids

        # calculate the coordinate matrix for each RBF anchor points
        self.cal_rbf_anchor_coord_mat()

        self.pos_enc_output_dim = self.num_rbf_anchor_pts

    def _random_sampling(self, item_tuple, num_sample):
        """
        poi_type_tuple: (Type1, Type2,...TypeM)
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
            # print("np.shape(coords)",np.shape(coords))
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
        coords_mat = np.expand_dims(coords_mat, axis=2)
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts, 2)
        coords_mat = np.repeat(coords_mat, self.num_rbf_anchor_pts, axis=2)
        # compute (deltaX, deltaY) between each point and each RBF anchor points
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts, 2)
        coords_mat = coords_mat - self.rbf_coords_mat
        # coords_mat: shape (batch_size, num_context_pt, num_rbf_anchor_pts=pos_enc_output_dim)
        coords_mat = np.sum(np.power(coords_mat, 2), axis=3)
        if self.rbf_kernel_size_ratio > 0:
            spr_embeds = np.exp(
                (-1 * coords_mat) / (2.0 * np.power(self.rbf_kernel_size_mat, 2))
            )
        else:
            # spr_embeds: shape (batch_size, num_context_pt, num_rbf_anchor_pts=pos_enc_output_dim)
            spr_embeds = np.exp(
                (-1 * coords_mat) / (2.0 * np.power(self.rbf_kernel_size, 2))
            )
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt=1, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds


class RBFSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        train_locs,
        model_type,
        coord_dim=2,
        device="cuda",
        num_rbf_anchor_pts=100,
        rbf_kernel_size=10e2,
        rbf_kernel_size_ratio=0.0,
        max_radius=10000,
        rbf_anchor_pt_ids=None,
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="RBFSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.train_locs = train_locs
        self.model_type = model_type
        self.num_rbf_anchor_pts = num_rbf_anchor_pts
        self.rbf_kernel_size = rbf_kernel_size
        self.rbf_kernel_size_ratio = rbf_kernel_size_ratio
        self.max_radius = max_radius
        self.rbf_anchor_pt_ids = rbf_anchor_pt_ids
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection
        self.ffn_context_str = ffn_context_str

        self.position_encoder = RBFSpatialRelationPositionEncoder(
            model_type=model_type,
            train_locs=train_locs,
            coord_dim=coord_dim,
            num_rbf_anchor_pts=num_rbf_anchor_pts,
            rbf_kernel_size=rbf_kernel_size,
            rbf_kernel_size_ratio=rbf_kernel_size_ratio,
            max_radius=max_radius,
            rbf_anchor_pt_ids=rbf_anchor_pt_ids,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=self.ffn_skip_connection,
            context_str=self.ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class SphereSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_pos_enc_output_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_pos_enc_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(
            3 * self.frequency_num
        )  # int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # convert to radius
        spr_embeds = spr_embeds * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # spr_embeds_: shape (batch_size, num_context_pt, 1, frequency_num, 3)
        spr_embeds_ = np.concatenate(
            [lat_sin, lat_cos * lon_cos, lat_cos * lon_sin], axis=-1
        )

        # # make sinuniod function
        # # sin for 2i, cos for 2i+1
        # # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=pos_enc_output_dim)
        # spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        # spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, frequency_num*3)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)

        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        return spr_embeds


class SphereSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="SphereSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = SphereSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class SphereGirdSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY), enco
    de them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_pos_enc_output_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_pos_enc_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(
            6 * self.frequency_num
        )  # int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # convert to radius
        spr_embeds = spr_embeds * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # spr_embeds_: shape (batch_size, num_context_pt, 1, frequency_num, 6)
        spr_embeds_ = np.concatenate(
            [lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_cos, lat_cos * lon_sin],
            axis=-1,
        )

        # (batch_size, num_context_pt, 2*frequency_num*6)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        return spr_embeds


class SphereGirdSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        device="cuda",
        freq_init="geometric",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="SphereGirdSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = SphereGirdSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class SphereMixScaleSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_pos_enc_output_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_pos_enc_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(
            5 * self.frequency_num
        )  # int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # convert to radius
        coords_mat = coords_mat * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_single_sin = np.sin(lat_single)
        lat_single_cos = np.cos(lat_single)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # spr_embeds_: shape (batch_size, num_context_pt, 1, frequency_num, 3)
        spr_embeds_ = np.concatenate(
            [
                lat_sin,
                lat_cos * lon_single_cos,
                lat_single_cos * lon_cos,
                lat_cos * lon_single_sin,
                lat_single_cos * lon_sin,
            ],
            axis=-1,
        )

        # (batch_size, num_context_pt, frequency_num*3)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)

        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        return spr_embeds


class SphereMixScaleSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="SphereMixScaleSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = SphereMixScaleSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class SphereGridMixScaleSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_output_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_output_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(
            8 * self.frequency_num
        )  # int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        if self.freq_init == "random":
            # the frequence we use for each block, alpha in ICLR paper
            # self.freq_list shape: (frequency_num)
            self.freq_list = (
                np.random.random(size=[self.frequency_num]) * self.max_radius
            )
        elif self.freq_init == "geometric":
            self.freq_list = []
            for cur_freq in range(self.frequency_num):
                base = 1.0 / (
                    np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
                )
                self.freq_list.append(base)

            self.freq_list = np.asarray(self.freq_list)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # convert to radius
        coords_mat = coords_mat * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_single_sin = np.sin(lat_single)
        lat_single_cos = np.cos(lat_single)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # spr_embeds_: shape (batch_size, num_context_pt, 1, frequency_num, 3)
        spr_embeds_ = np.concatenate(
            [
                lat_sin,
                lat_cos,
                lon_sin,
                lon_cos,
                lat_cos * lon_single_cos,
                lat_single_cos * lon_cos,
                lat_cos * lon_single_sin,
                lat_single_cos * lon_sin,
            ],
            axis=-1,
        )

        # (batch_size, num_context_pt, frequency_num*3)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        return spr_embeds


class SphereGridMixScaleSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="SphereGridMixScaleSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = SphereGridMixScaleSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class DFTSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.pos_enc_output_dim = self.cal_input_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (pos_enc_output_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return (
            self.frequency_num * 4 + 4 * self.frequency_num * self.frequency_num
        )  # int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        if self.freq_init == "random":
            # the frequence we use for each block, alpha in ICLR paper
            # self.freq_list shape: (frequency_num)
            self.freq_list = (
                np.random.random(size=[self.frequency_num]) * self.max_radius
            )
        elif self.freq_init == "geometric":
            self.freq_list = []
            for cur_freq in range(self.frequency_num):
                base = 1.0 / (
                    np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
                )
                self.freq_list.append(base)

            self.freq_list = np.asarray(self.freq_list)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # convert to radius
        coords_mat = coords_mat * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_single_sin = np.sin(lat_single)
        lat_single_cos = np.cos(lat_single)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # lon_feq: shape (batch_size, num_context_pt, 1, 2*frequency_num, 1)
        lon_feq = np.concatenate([lon_sin, lon_cos], axis=-2)

        # lat_feq: shape (batch_size, num_context_pt, 1, 2*frequency_num, 1)
        lat_feq = np.concatenate([lat_sin, lat_cos], axis=-2)

        # lat_feq_t: shape (batch_size, num_context_pt, 1, 1, 2*frequency_num)
        lat_feq_t = lat_feq.transpose(0, 1, 2, 4, 3)

        # coord_freq: shape (batch_size, num_context_pt, 1, 2*frequency_num, 2*frequency_num)
        coord_freq = np.einsum("abcde,abcek->abcdk", lon_feq, lat_feq_t)

        # coord_freq_: shape (batch_size, num_context_pt, 2*frequency_num * 2*frequency_num)
        coord_freq_ = np.reshape(coord_freq, (batch_size, num_context_pt, -1))

        # coord_1: shape (batch_size, num_context_pt, 1, frequency_num, 4)
        coord_1 = np.concatenate([lat_sin, lat_cos, lon_sin, lon_cos], axis=-1)

        # coord_1_: shape (batch_size, num_context_pt, frequency_num * 4)
        coord_1_ = np.reshape(coord_1, (batch_size, num_context_pt, -1))

        # spr_embeds_: shape (batch_size, num_context_pt, frequency_num * 4 + 4*frequency_num ^^2)
        spr_embeds_ = np.concatenate([coord_freq_, coord_1_], axis=-1)

        # # make sinuniod function
        # # sin for 2i, cos for 2i+1
        # # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=pos_enc_output_dim)
        # spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        # spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, frequency_num*3)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        # return sprenc
        return spr_embeds


class DFTSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="DFTSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = DFTSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            freq_init=freq_init,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class RFFSpatialRelationPositionEncoder(PositionEncoder):
    """
    Random Fourier Feature
    Based on paper - Random Features for Large-Scale Kernel Machines
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        rbf_kernel_size=1.0,
        extent=None,
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            extent: (x_min, x_max, y_min, y_max)
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: here, we understand it as the RFF embeding dimension before FNN
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.frequency_num = frequency_num
        self.rbf_kernel_size = rbf_kernel_size
        self.extent = extent

        self.generate_direction_vector()

        self.pos_enc_output_dim = self.frequency_num

    def generate_direction_vector(self):
        """
        Generate K direction vector (omega) and shift vector (b)
        Return:
            dirvec: shape (coord_dim, frequency_num), omega in the paper
            shift: shape (frequency_num), b in the paper
        """
        # mean and covarance matrix of the Gaussian distribution
        self.mean = np.zeros(self.coord_dim)
        self.cov = np.diag(np.ones(self.coord_dim) * self.rbf_kernel_size)
        # dirvec: shape (coord_dim, frequency_num), omega in the paper
        dirvec = np.transpose(
            np.random.multivariate_normal(self.mean, self.cov, self.frequency_num)
        )
        self.dirvec = torch.nn.Parameter(torch.FloatTensor(dirvec), requires_grad=False)
        self.register_parameter("dirvec", self.dirvec)

        # shift: shape (frequency_num), b in the paper
        shift = np.random.uniform(0, 2 * np.pi, self.frequency_num)
        self.shift = torch.nn.Parameter(torch.FloatTensor(shift), requires_grad=False)
        self.register_parameter("shift", self.shift)

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = coord_normalize(coords_mat, self.extent)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = torch.FloatTensor(coords_mat).to(self.device)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = frequency_num)
        spr_embeds = torch.matmul(coords_mat, self.dirvec)

        spr_embeds = torch.cos(spr_embeds + self.shift) * np.sqrt(
            2.0 / self.pos_enc_output_dim
        )

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = frequency_num)
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = frequency_num)
        spr_embeds = self.make_output_embeds(coords)

        return spr_embeds


class RFFSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        frequency_num=16,
        rbf_kernel_size=1.0,
        extent=None,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="RFFSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.rbf_kernel_size = rbf_kernel_size
        self.extent = extent
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = RFFSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            frequency_num=frequency_num,
            rbf_kernel_size=rbf_kernel_size,
            extent=extent,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class GridLookupSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY),
    divide the space into grids, each point is using the grid embedding it falls into

    """

    def __init__(
        self,
        spa_embed_dim,
        model_type="global",
        max_radius=None,
        coord_dim=2,
        interval=1000000,
        extent=[-180, 180, -90, 90],
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            interval: the cell size in X and Y direction
            extent: (left, right, bottom, top)
                "global": the extent of the study area (-1710000, -1690000, 1610000, 1640000)
                "relative": the extent of the relative context
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.spa_embed_dim = spa_embed_dim
        self.interval = interval
        self.model_type = model_type
        self.max_radius = max_radius   
        assert extent[0] < extent[1]
        assert extent[2] < extent[3]

        self.extent = self.get_spatial_context(model_type, extent, max_radius)
        self.make_grid_embedding(self.interval, self.extent)

        self.pos_enc_output_dim = spa_embed_dim

    def get_spatial_context(self, model_type, extent, max_radius):
        if model_type == "global":
            extent = extent
        elif model_type == "relative":
            extent = (
                -max_radius - 500,
                max_radius + 500,
                -max_radius - 500,
                max_radius + 500,
            )
        return extent

    def make_grid_embedding(self, interval, extent):
        self.num_col = int(math.ceil(float(extent[1] - extent[0]) / interval))
        self.num_row = int(math.ceil(float(extent[3] - extent[2]) / interval))

        self.embedding = torch.nn.Embedding(
            self.num_col * self.num_row, self.spa_embed_dim
        )
        self.embedding.weight.data.normal_(0, 1.0 / self.spa_embed_dim)

    def make_output_embeds(self, coords):
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

        # x or y: shape (batch_size, num_context_pt)
        x = coords_mat[:, :, 0]
        y = coords_mat[:, :, 1]

        col = np.floor((x - self.extent[0]) / self.interval)
        row = np.floor((y - self.extent[2]) / self.interval)

        # make sure each row/col index is within range
        col = np.clip(col, 0, self.num_col - 1)
        row = np.clip(row, 0, self.num_row - 1)

        # index_mat: shape (batch_size, num_context_pt)
        index_mat = (row * self.num_col + col).astype(int)
        # index_mat: shape (batch_size, num_context_pt)
        index_mat = torch.LongTensor(index_mat).to(self.device)  

        spr_embeds = self.embedding(torch.autograd.Variable(index_mat))
        # spr_embeds: shape (batch_size, num_context_pt, spa_embed_dim)
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # spr_embeds: shape (batch_size, num_context_pt, spa_embed_dim)
        spr_embeds = self.make_output_embeds(coords).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        return spr_embeds


class GridLookupSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        extent,
        interval,
        coord_dim=2,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="GridLookupSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.extent = extent
        self.interval = (interval,)
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = GridLookupSpatialRelationPositionEncoder(
            spa_embed_dim=spa_embed_dim,
            model_type="global",
            coord_dim=coord_dim,
            extent=extent,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc


class AodhaFFTSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (deltaX,deltaY),
    divide the space into grids, each point is using the grid embedding it falls into

    spa_enc_type == "geo_net_fft"
    """

    def __init__(
        self,
        extent,
        coord_dim=2,
        do_pos_enc=False,
        do_global_pos_enc=True,
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
                if do_pos_enc == False:
                    coord_dim is the computed loc_feat according to get_model_input_feat_dim()
            extent: (x_min, x_max, y_min, y_max)
            do_pos_enc:     True  - we normalize the lat/lon, and [sin(pi*x), cos(pi*x), sin(pi*y), cos(pi*y)]
                            False - we assume the input is prenormalized coordinate features
            do_global_pos_enc:  if do_pos_enc == True:
                            True - lon/180 and lat/90
                            False - min-max normalize based on extent
            f_act: the final activation function, relu ot none
        """
        super().__init__(coord_dim=coord_dim, device=device)
        self.extent = extent
        self.coord_dim = coord_dim

        self.do_pos_enc = do_pos_enc
        self.do_global_pos_enc = do_global_pos_enc
        self.pos_enc_output_dim = 4

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
            # coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for AodhaSpatialRelationEncoder")

        assert coords.shape[-1] == 2
        # coords: shape (batch_size, num_context_pt, 2)
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = coord_normalize(
            coords, self.extent, do_global=self.do_global_pos_enc
        )

        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        loc_sin = np.sin(math.pi * coords_mat)
        loc_cos = np.cos(math.pi * coords_mat)
        # spr_embeds: shape (batch_size, num_context_pt, 4)
        spr_embeds = np.concatenate((loc_sin, loc_cos), axis=-1)

        # spr_embeds: shape (batch_size, num_context_pt, 4)
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        if self.do_pos_enc:
            # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
            spr_embeds = self.make_output_embeds(coords)
            # assert self.pos_enc_output_dim == np.shape(spr_embeds)[2]
        else:
            # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
            spr_embeds = coords

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds


class AodhaFFNSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        extent,
        coord_dim=2,
        do_pos_enc=False,
        do_global_pos_enc=True,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="AodhaFFTSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.extent = extent
        self.do_pos_enc = do_pos_enc
        self.do_global_pos_enc = do_global_pos_enc
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection

        self.position_encoder = AodhaFFTSpatialRelationPositionEncoder(
            coord_dim=coord_dim,
            extent=extent,
            do_pos_enc=do_pos_enc,
            do_global_pos_enc=do_global_pos_enc,
            device=device,
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc

class SphericalHarmonicsSpatialRelationPositionEncoder(PositionEncoder):
    """
    Given a list of (lon,lat), convert them to (x,y,z), and then encode them using the MLP

    """

    def __init__(self, coord_dim=2, legendre_poly_num=8, device="cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            extent: (x_min, x_max, y_min, y_max)
        """
        super().__init__(coord_dim=coord_dim, device=device)

        self.legendre_poly_num = legendre_poly_num
        self.pos_enc_output_dim = legendre_poly_num**2

    def make_output_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for SphericalHarmonicsSpatialRelationEncoder"
            )

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # lon: (batch_size, num_context_pt, 1), convert from degree to radius
        lon = np.deg2rad(coords_mat[:, :, :1])
        # lat: (batch_size, num_context_pt, 1), convert from degree to radius
        lat = np.deg2rad(coords_mat[:, :, 1:])

        # spr_embeds: (batch_size, num_context_pt, legendre_poly_num**2)
        spr_embeds = get_positional_encoding(lon, lat, self.legendre_poly_num)
        return spr_embeds.reshape((-1, self.pos_enc_output_dim))

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

        # spr_embeds: (batch_size, num_context_pt, 3)
        spr_embeds = self.make_output_embeds(coords)

        # spr_embeds: shape (batch_size, num_context_pt, pos_enc_output_dim = 3)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return spr_embeds

class SphericalHarmonicsSpatialRelationLocationEncoder(LocationEncoder):
    def __init__(
        self,
        spa_embed_dim,
        coord_dim=2,
        legendre_poly_num=8,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="SphericalHarmonicsSpatialRelationEncoder",
    ):
        super().__init__(spa_embed_dim, coord_dim, device)
        self.ffn_act = ffn_act
        self.ffn_num_hidden_layers = ffn_num_hidden_layers
        self.ffn_dropout_rate = ffn_dropout_rate
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_use_layernormalize = ffn_use_layernormalize
        self.ffn_skip_connection = ffn_skip_connection
        self.ffn_context_str = ffn_context_str

        self.position_encoder = SphericalHarmonicsSpatialRelationPositionEncoder(
            coord_dim=coord_dim, legendre_poly_num=legendre_poly_num, device=device
        )
        self.ffn = MultiLayerFeedForwardNN(
            input_dim=self.position_encoder.pos_enc_output_dim,
            output_dim=self.spa_embed_dim,
            num_hidden_layers=self.ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=self.ffn_hidden_dim,
            activation=self.ffn_act,
            use_layernormalize=self.ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str=ffn_context_str,
        )

    def forward(self, coords):
        spr_embeds = self.position_encoder(coords)
        sprenc = self.ffn(spr_embeds)

        return sprenc
