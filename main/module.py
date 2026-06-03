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
    elif activation == 'gelu':
        return nn.GELU()
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
        nn.init.xavier_uniform_(self.linear.weight)

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


class Sine(nn.Module):
    """Sine activation with a configurable angular frequency ``w0`` (used by SIREN layers)."""

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """A single SIREN layer with optional residual connection and H-SIREN activation.

    When ``residual_connections`` is True, the pre-activation vector (``gaussian``) of the
    previous layer is averaged into the current pre-activation vector (only when their shapes
    match), keeping the residual stream inside SIREN's distributional constraints.

    When ``h_siren`` is True, the first layer uses the H-SIREN activation ``sin(sinh(2x))``
    instead of a plain ``sin(x)``: passing the pre-activation through ``sinh`` before the sine
    widens the supported frequency set and reduces SIREN's tendency to over-smooth. Introduced
    in H-SIREN (Gao & Jaiman, "H-SIREN: Improving implicit neural representations with
    hyperbolic periodic functions", 2024, https://arxiv.org/abs/2410.04716). It only affects the
    first layer (``is_first``).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
        residual_connections=False,
        h_siren=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout
        self.h_siren = h_siren

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

        self.residual_connections = residual_connections

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, prev_gaussian=None):
        # Apply base linear transform
        out = F.linear(x, self.weight, self.bias)
        # Apply dropout (p=0.0 is a no-op; not recommended for reconstruction)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # If using ReSIREN, merge with the previous layer's gaussian (pre-activation)
        if (
            self.residual_connections
            and prev_gaussian is not None
            and out.shape == prev_gaussian.shape
        ):
            out = (out + prev_gaussian) / 2
        # Save the gaussian to pass it on
        gaussian = out
        # Apply the additional transformation for H-SIREN
        if self.h_siren and self.is_first:
            out = torch.sinh(2 * out)
        # Apply the activation (for nearly all layers Sine)
        out = self.activation(out)
        return out, gaussian


class SirenNet(nn.Module):
    """Sinusoidal Representation Network (SIREN), optionally with residual connections (ReSIREN).

    SIREN was introduced in:

        Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions",
        NeurIPS 2020. https://arxiv.org/abs/2006.09661

    This is a reusable, position-encoding-agnostic network (the SIREN counterpart of
    MultiLayerFeedForwardNN): it maps an input embedding to an output embedding through
    sinusoidal layers. Pair it with any PositionEncoder (e.g. the Direct encoding used by
    ReSiren, or the spherical-harmonics encoding) to build a location encoder.

    Adapted from the SatCLIP location encoder
    (https://github.com/microsoft/satclip/blob/main/satclip/location_encoder.py) and extended
    with residual connections + the H-SIREN first-layer activation, as introduced in:

        Dollinger et al., "Climplicit: Climatic Implicit Embeddings for Global Ecological
        Tasks", ICLR 2025 Workshops. https://arxiv.org/abs/2504.05089

    A plain SIREN requires a specific pre-activation distribution between layers, so naive
    residual connections on the post-activation outputs break that constraint. ReSiren instead
    connects the pre-activation hidden vectors (the well-behaved "gaussian" before the sine),
    which keeps the residual stream within SIREN's distributional constraints and lets the
    network scale to many more layers.

    Args:
        dim_in (int): input dimension.
        dim_hidden (int): hidden dimension of every SIREN layer.
        dim_out (int): output (embedding) dimension.
        num_layers (int): number of hidden SIREN layers (a final linear SIREN layer is added).
        w0 (float): angular frequency for the hidden layers.
        w0_initial (float): angular frequency for the first layer (SIREN's larger init).
        residual_connections (bool): turn the network into a ReSIREN model.
        h_siren (bool): use the H-SIREN first-layer activation ``sin(sinh(2x))`` instead of a
            plain sine on the first layer (Gao & Jaiman, 2024, arXiv:2410.04716). Widens the
            supported frequency set and reduces over-smoothing.
        return_hidden_embs (list[int] | None): if set, append the embeddings of those hidden
            layer indices to the output (multi-resolution embeddings).
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        dropout=0.0,
        residual_connections=False,
        h_siren=False,
        return_hidden_embs=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.return_hidden_embs = return_hidden_embs

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    h_siren=h_siren,
                    dropout=dropout,
                    residual_connections=residual_connections,
                )
            )

        final_activation = nn.Identity() if not final_activation else final_activation
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=0.0,
        )

    def forward(self, x, mods=None):
        res = []
        gaussian = None
        # Pass the gaussian (pre-activation vector) between layers as the residual connection
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, gaussian = layer(x, gaussian)
            if self.return_hidden_embs is not None:
                if i in self.return_hidden_embs:
                    res.append(x)

        x, _ = self.last_layer(x, gaussian)

        if len(res) > 0:
            res.append(x)
            x = torch.cat(res, dim=-1)
        return x


class PositionEncoder(nn.Module):
    def __init__(self, coord_dim=2, device="cuda"):
        super(PositionEncoder, self).__init__()
        self.coord_dim = coord_dim
        self.device = device
        self.pos_enc_output_dim = None  # self.cal_pos_enc_output_dim()

    def cal_pos_enc_output_dim(self):
        raise NotImplementedError(
            "The 'pos_enc_output_dim' property should be implemented by subclasses."
        )

    def forward(self, coords):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")


class LocationEncoder(nn.Module):
    def __init__(self, spa_embed_dim, coord_dim=2, device="cuda"):
        super(LocationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.device = device

    def vil_attribute(self):
        raise NotImplementedError("Subclasses must implement this property.")

    def forward(self, coords):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")
