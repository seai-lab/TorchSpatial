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

- **Purpose**: Processes input coordinates through the location encoder to produce final spatial embeddings.
- **Parameters**: 
   - ``coords`` (List or np.ndarray): Coordinates to process, expected to be in the form ``(batch_size, num_context_pt, coord_dim)``.
- **Returns**:
   - ``sprenc`` (Tensor): Spatial relation embeddings with a shape of ``(batch_size, num_context_pt, spa_embed_dim)``.

:class:`NERFSpatialRelationPositionEncoder`
============================================

Features
--------

.. image:: ../images/NeRF.png

Configuration Parameters
------------------------

- **coord_dim**: Dimensionality of the space being encoded (e.g., 2D, 3D).
- **frequency_num**: Number of different sinusoidal frequencies used to encode spatial differences.
- **freq_init**: Frequency initialization method, set to 'nerf' for NeRF-based encoding.
- **device**: Specifies the computational device, e.g., 'cuda' for GPU acceleration.

Methods
--------

.. method:: cal_freq_list()

- **Purpose**: Calculates the list of frequencies used for the sinusoidal encoding based on the NeRF methodology, using an exponential scaling of frequencies.
- **Modifies**:
    - Internal frequency list based on the specified initialization method.

.. method:: cal_freq_mat()

- **Purpose**: Creates a frequency matrix to be used in the encoding process.
- **Modifies**:
    - Internal frequency matrix to match the dimensions required for vectorized operations.

.. method:: make_output_embeds(coords)

- **Purpose**: Processes a batch of coordinates and converts them into spatial relation embeddings.
- **Parameters**:
    - ``coords``: Batch of geographic coordinates.
- **Returns**:
    - Batch of spatial relation embeddings in high-dimensional space.

Implementation Details
----------------------

- Converts longitude and latitude to radians, then to Cartesian coordinates assuming a unit sphere.
- Applies sinusoidal functions to these Cartesian coordinates, scaled by the computed frequencies.
- Outputs high-dimensional embeddings based on these sinusoidally encoded coordinates.

Usage Example
-------------

.. code-block:: python

    # Initialize the encoder
    encoder = NERFSpatialRelationLocationEncoder(
        spa_embed_dim=64,
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
        ffn_context_str="NERFSpatialRelationEncoder"
    )

    # Sample coordinates
    coords = np.array([[34.0522, -118.2437],..., [40.7128, -74.0060]])  # Example: [latitude, longitude]

    # Generate spatial embeddings
    embeddings = encoder.forward(coords)
