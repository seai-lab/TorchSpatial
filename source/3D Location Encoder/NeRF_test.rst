*NeRF*
======

:class:`NERFSpatialRelationLocationEncoder`
============================================

Overview
--------

The :class:`NERFSpatialRelationLocationEncoder` is designed to compute spatial embeddings from coordinate data using a Neural Radiance Field (NeRF) based encoding approach. This encoder integrates a position encoding strategy, leveraging a :class:`NERFSpatialRelationPositionEncoder`, and further processes the encoded positions through a customizable multi-layer feed-forward neural network.

Features
--------

- **Position Encoding** (``self.position_encoder``): Utilizes the :class:`NERFSpatialRelationPositionEncoder` to encode spatial differences (latitude, longitude) using NeRF-inspired sinusoidal functions.
- **Feed-Forward Neural Network** (``self.ffn``): Transforms the position-encoded data through a series of feed-forward layers to produce high-dimensional spatial embeddings.

Configuration Parameters
------------------------

- ``spa_embed_dim``: Dimensionality of the spatial embedding output.
- ``coord_dim``: Dimensionality of the coordinate space (e.g., 2D, 3D).
- ``device``: Computation device (e.g., 'cuda' for GPU).
- ``frequency_num``: Number of frequency components used in positional encoding.
- ``freq_init``: Initial setting for frequency calculation, set to 'nerf' for NeRF-specific frequency calculations.
- ``ffn_act``: Activation function for the feed-forward layers.
- ``ffn_num_hidden_layers``: Number of hidden layers in the feed-forward network.
- ``ffn_dropout_rate``: Dropout rate used in the feed-forward network.
- ``ffn_hidden_dim``: Dimension of each hidden layer in the feed-forward network.
- ``ffn_use_layernormalize``: Flag to enable layer normalization in the feed-forward network.
- ``ffn_skip_connection``: Flag to enable skip connections in the feed-forward network.
- ``ffn_context_str``: Context string for debugging and detailed logging within the network.

Methods
--------

.. method:: forward(coords)

   **Purpose**: Processes input coordinates through the location encoder to produce final spatial embeddings.

   **Parameters**:
   - ``coords`` (List or np.ndarray): Coordinates to process, expected to be in the form ``(batch_size, num_context_pt, coord_dim)``.

   **Returns**:
   - ``sprenc`` (Tensor): Spatial relation embeddings with a shape of ``(batch_size, num_context_pt, spa_embed_dim)``.
