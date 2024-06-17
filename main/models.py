import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np

### Feature Encoder for SustainBench regression model
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
        elif isinstance(coords, torch.Tensor):
            assert self.coord_dim == coords.shape[2]
            coords = coords.tolist()
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


class SustainBenchRegressNet(nn.Module):
    def __init__(self, train_dataset, device, params, loc_enc):
        super(SustainBenchRegressNet, self).__init__()
        self.position_encoder = RBFFeaturePositionEncoder(train_locs=train_dataset, num_rbf_anchor_pts=params["num_rbf_anchor_pts"], rbf_kernel_size=params["rbf_kernel_size"], device=device)
        self.loc_enc = loc_enc

        if params['dataset'] == 'sustainbench_under5_mort':
            self.img_model = nn.Sequential(
                nn.Linear(self.position_encoder.pos_enc_output_dim, 980),
                nn.LeakyReLU(),
                nn.Linear(980, params["embed_dim_before_regress"])
            )
            self.ffn = nn.Sequential(
                nn.Linear(in_features= params["embed_dim_before_regress"]*2, out_features=979),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=979, out_features=919),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=919, out_features=523),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=523, out_features=1),
            )
        elif params['dataset'] == 'sustainbench_women_bmi':
            self.img_model = nn.Sequential(
                nn.Linear(self.position_encoder.pos_enc_output_dim, 744),
                nn.ReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(744, params["embed_dim_before_regress"]),
            )
            self.ffn = nn.Sequential(
                nn.Linear(in_features= params["embed_dim_before_regress"]*2, out_features=707),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=707, out_features=919),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=919, out_features=523),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=523, out_features=1),
            )

    def forward(self, img_feats, locs):
        '''
        Args:
            feats: shape [batch_size, 1]
            locs: shape[batch_size, 2]
        '''
        feat_position = self.position_encoder(img_feats)
        cnn_embed = self.img_model(feat_position)

        loc_embed = torch.squeeze(self.loc_enc(locs), dim=1)

        input_embed = torch.cat([cnn_embed, loc_embed], dim=-1)
        outputs = self.ffn(input_embed)

        return outputs


class MosaiksRegressNet(nn.Module):
    def __init__(self, params, device, loc_enc, dropout_p=0.2, input_dim=32, hidden_dim=256):
        super(MosaiksRegressNet, self).__init__()
        self.device = device
        self.loc_enc = loc_enc
        self.input_dim = input_dim

        if params['dataset'] == 'mosaiks_elevation':
            self.cnn_ffn = nn.Sequential(
                nn.Linear(in_features=2048, out_features=862),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=862, out_features=846),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=846, out_features=input_dim),
            )
            self.regress_ffn = nn.Sequential(
                nn.Linear(in_features=input_dim*2, out_features=893),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=893, out_features=881),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=881, out_features=510),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_features=510, out_features=1),
            )
        elif params['dataset'] == 'mosaiks_forest_cover':
            self.cnn_ffn = nn.Sequential(
                nn.Linear(in_features=2048, out_features=533),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=533, out_features=input_dim),
            )
            self.regress_ffn = nn.Sequential(
                nn.Linear(in_features=input_dim*2, out_features=847),
                nn.ReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=847, out_features=787),
                nn.ReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_features=787, out_features=1),
            )

        elif params['dataset'] == 'mosaiks_nightlights':
            self.cnn_ffn = nn.Sequential(
                nn.Linear(in_features=2048, out_features=346),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=346, out_features=682),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=682, out_features=73),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=73, out_features=input_dim),
            )
            self.regress_ffn = nn.Sequential(
                nn.Linear(in_features=input_dim*2, out_features=746),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=746, out_features=416),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=416, out_features=257),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=257, out_features=1),
            )

        elif params['dataset'] == 'mosaiks_population':
            self.cnn_ffn = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_features=1024, out_features=input_dim),
            )
            self.regress_ffn = nn.Sequential(
                nn.Linear(in_features=input_dim*2, out_features=16),
                nn.LeakyReLU(),  # nn.Tanh(),ReLU()
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_features=16, out_features=1),
            )


    def forward(self, img_feats, locs):
        '''
        Args:
            img_feats: shape [batch_size, 2048]
            locs: shape[batch_size, 2]
        '''
        img_feats = img_feats.to(self.device)  # Move img_feats to the same device as the model
        locs = locs.to(self.device)  # Move locs to the same device as the model

        img_feats = img_feats.float()
        cnn_embed = self.cnn_ffn(img_feats)
        loc_embed = torch.squeeze(self.loc_enc(locs), dim=1)

        input_embed = torch.cat([cnn_embed, loc_embed], dim=-1)
        #input_embed = cnn_embed * loc_embed
        outputs = self.regress_ffn(input_embed)

        return outputs


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=1):
        '''
        Args:
            num_inputs: input embedding dimention
            num_classes: number of categories we want to classify
            num_filts: hidden embedding dimention
        '''
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.num_filts = num_filts
        self.num_classes = num_classes
        self.num_users = num_users
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        if self.num_users is not None:
            self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))

    def forward(self, x, class_of_interest=None, return_feats=False):
        '''
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2 or 3 or ...)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        '''
        x = x.to(self.feats[0].weight.device) 
        loc_emb = self.feats(x)
        if return_feats:
            # loc_emb: (batch_size, num_filts)
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        # return (batch_size, num_classes)
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        '''
        Return:
            shape (batch_size)
        '''
        # note: self.class_emb.weight shape (num_classes, num_filts)
        class_weights = self.class_emb.weight[class_of_interest, :].to(x.device)

        # Perform matrix multiplication
        if self.inc_bias:
            return torch.matmul(x, class_weights) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, class_weights)


class TangNet(nn.Module):
    def __init__(self, ip_loc_dim, feats_dim, loc_dim, num_classes, use_loc):
        super(TangNet, self).__init__()
        '''
        ip_loc_dim: the number of grids to be consider
        feats_dim: the image embedding dimention
        loc_dim: the dimention of location featire
        num_classes: the number of image category
        use_loc: True/False
        '''
        self.use_loc = use_loc
        # the location embedding matrix
        self.fc_loc = nn.Linear(ip_loc_dim, loc_dim)
        if self.use_loc:
            self.fc_class = nn.Linear(feats_dim + loc_dim, num_classes)
        else:
            self.fc_class = nn.Linear(feats_dim, num_classes)

    def forward(self, loc, net_feat):
        '''
        Args:
            locs: one hot vector of one location
            net_feat: the image features
        '''
        if self.use_loc:
            x = torch.sigmoid(self.fc_loc(loc))
            x = self.fc_class(torch.cat((x, net_feat), 1))
        else:
            x = self.fc_class(net_feat)
        return F.log_softmax(x, dim=1)


class LocationEncoder(nn.Module):
    def __init__(self, spa_enc, num_inputs, num_classes, num_filts, num_users=1):
        '''
        Args:
            spa_enc: the spatial encoder
            num_inputs: input embedding dimention
            num_classes: number of categories we want to classify
            num_filts: hidden embedding dimention
        '''
        super(LocationEncoder, self).__init__()
        self.spa_enc = spa_enc
        self.inc_bias = False
        self.num_filts = num_filts
        self.num_classes = num_classes
        self.num_users = num_users
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        if self.num_users is not None:
            self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

    def forward(self, x, class_of_interest=None, return_feats=False):
        '''
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        '''
        # loc_feat: (batch_size, 1, input_loc_dim = 2)
        loc_feat = torch.unsqueeze(x, dim=1)
        loc_feat = loc_feat.cpu().data.numpy()

        # loc_embed: torch.Tensor(), (batch_size, 1, spa_embed_dim = num_filts)
        loc_embed = self.spa_enc(loc_feat)
        # loc_emb: torch.Tensor(), (batch_size, spa_embed_dim = num_filts)
        loc_emb = loc_embed.squeeze(1)
        if return_feats:
            # loc_emb: (batch_size, num_filts)
            return loc_emb
        if class_of_interest is None:
            # class_pred: (batch_size, num_classes)
            class_pred = self.class_emb(loc_emb)
        else:
            # class_pred: shape (batch_size)
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        '''
        Args:
            x: (batch_size, num_filts)
        Return:
            shape (batch_size)
        '''
        # note: self.class_emb.weight shape (num_classes, num_filts)
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class LocationImageEncoder(nn.Module):
    def __init__(self, loc_enc, train_loss, unsuper_loss="none", cnn_feat_dim=2048, spa_enc_type="sphere"):
        '''
        Args:
            loc_enc: LocationEncoder() or FCNet()
        '''
        super(LocationImageEncoder, self).__init__()
        self.loc_enc = loc_enc
        if spa_enc_type in ["geo_net"]:
            self.spa_enc = loc_enc
        else:
            self.spa_enc = loc_enc.spa_enc
        self.inc_bias = loc_enc.inc_bias
        self.class_emb = loc_enc.class_emb
        self.user_emb = loc_enc.user_emb

        self.cnn_feat_dim = cnn_feat_dim
        self.loc_emb_dim = loc_enc.num_filts

        if unsuper_loss == "none":
            return
        elif unsuper_loss == "l2regress":
            self.loc_dec = nn.Linear(
                in_features=self.loc_emb_dim, out_features=self.cnn_feat_dim, bias=True)
        elif "imgcontloss" in unsuper_loss or "contsoftmax" in unsuper_loss:
            self.img_dec = nn.Linear(
                in_features=self.cnn_feat_dim, out_features=self.loc_emb_dim, bias=True)
        else:
            raise Exception(f"Unknown unsuper_loss={unsuper_loss}")

    def forward(self, x, class_of_interest=None, return_feats=False):
        '''
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        '''
        return self.loc_enc.forward(x, class_of_interest, return_feats)

    def eval_single_class(self, x, class_of_interest):
        '''
        Args:
            x: (batch_size, num_filts)
        Return:
            shape (batch_size)
        '''
        return self.loc_enc.eval_single_class(x, class_of_interest)
