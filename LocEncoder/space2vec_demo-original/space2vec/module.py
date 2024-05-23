import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import torch.utils.data
import math


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim,
                 dropout_rate=None,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=''):
        '''

        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN

        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.

        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output

#     num_rbf_anchor_pts = 100, rbf_kernal_size = 10e2, frequency_num = 16,
# max_radius = 10000, dropout = 0.5, f_act = "sigmoid", freq_init = "geometric",
# num_hidden_layer = 3, hidden_dim = 128, use_layn = "F", skip_connection = "F", use_post_mat = "T"):
#     if use_layn == "T":
#         use_layn = True
#     else:
#         use_layn = False
#     if skip_connection == "T":
#         skip_connection = True
#     else:
#         skip_connection = False
#     if use_post_mat == "T":
#         use_post_mat = True
#     else:
#         use_post_mat = False


class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                 output_dim,
                 num_hidden_layers=0,
                 dropout_rate=None,
                 hidden_dim=-1,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=None):
        '''

        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN

        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                       output_dim=self.hidden_dim,
                                                       dropout_rate=self.dropout_rate,
                                                       activation=self.activation,
                                                       use_layernormalize=self.use_layernormalize,
                                                       skip_connection=self.skip_connection,
                                                       context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.

        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output


# from Presence-Only Geographical Priors for Fine-Grained Image Classification
# www.vision.caltech.edu/~macaodha/projects/geopriors

# class ResLayer(nn.Module):
#     def __init__(self, linear_size):
#         super(ResLayer, self).__init__()
#         self.l_size = linear_size
#         self.nonlin1 = nn.ReLU(inplace=True)
#         self.nonlin2 = nn.ReLU(inplace=True)
#         self.dropout1 = nn.Dropout()
#         self.w1 = nn.Linear(self.l_size, self.l_size)
#         self.w2 = nn.Linear(self.l_size, self.l_size)

#     def forward(self, x):
#         y = self.w1(x)
#         y = self.nonlin1(y)
#         y = self.dropout1(y)
#         y = self.w2(y)
#         y = self.nonlin2(y)
#         out = x + y

#         return out


# class FCNet(nn.Module):
#     # def __init__(self, num_inputs, num_classes, num_filts, num_users=1):
#     def __init__(self, num_inputs, num_filts, num_hidden_layers):
#         '''
#         Args:
#             num_inputs: input embedding diemntion
#             num_filts: hidden embedding dimention
#             num_hidden_layers: number of hidden layer
#         '''
#         super(FCNet, self).__init__()
#         # self.inc_bias = False
#         # self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
#         # self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

#         # self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
#         #                             nn.ReLU(inplace=True),
#         #                             ResLayer(num_filts),
#         #                             ResLayer(num_filts),
#         #                             ResLayer(num_filts),
#         #                             ResLayer(num_filts))
#         self.num_hidden_layers = num_hidden_layers
#         self.feats = nn.Sequential()
#         self.feats.add_module("ln_1", nn.Linear(num_inputs, num_filts))
#         self.feats.add_module("relu_1", nn.ReLU(inplace=True))
#         for i in range(num_hidden_layers):
#             self.feats.add_module("resnet_{}".format(i+1), ResLayer(num_filts))

#     # def forward(self, x, class_of_interest=None, return_feats=False):
#     def forward(self, x):
#         loc_emb = self.feats(x)
#         # if return_feats:
#         #     return loc_emb
#         # if class_of_interest is None:
#         #     class_pred = self.class_emb(loc_emb)
#         # else:
#         #     class_pred = self.eval_single_class(loc_emb, class_of_interest)

#         # return torch.sigmoid(class_pred)
#         return loc_emb

#     # def eval_single_class(self, x, class_of_interest):
#     #     if self.inc_bias:
#     #         return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
#     #     else:
#     #         return torch.matmul(x, self.class_emb.weight[class_of_interest, :])
