*Space2Vec-sphereM*
++++++++++++++++++++++++++++++++++++++

:class:`SphereMixScaleSpatialRelationLocationEncoder`
=====================================================

Overview
--------

The :class:`SphereMixScaleSpatialRelationLocationEncoder` is engineered to encode spatial relationships between locations using advanced position encoding techniques. It integrates the :class:`SphereMixScaleSpatialRelationPositionEncoder` for initial encoding and processes the results through a multi-layer feed-forward neural network to produce high-dimensional spatial embeddings.

Features
--------

- **Position Encoding** ``self.position_encoder``: Utilizes the :class:`SphereMixScaleSpatialRelationPositionEncoder` to encode spatial differences (deltaX, deltaY) using geometrically scaled sinusoidal functions.
- **Feed-Forward Neural Network** ``self.ffn``: Transforms position-encoded data through several neural network layers to produce high-dimensional spatial embeddings.

Configuration Parameters
------------------------

- ``spa_embed_dim``: The dimensionality of the output spatial embeddings.
- ``coord_dim``: The dimensionality of the coordinate space, typically 2D.
- ``device``: Specifies the computation device, e.g., 'cuda'.
- ``frequency_num``: Number of frequency components used in positional encoding.
- ``max_radius``: The largest spatial context radius the model can handle.
- ``min_radius``: The minimum radius, ensuring detailed capture at smaller scales.
- ``freq_init``: Initialization method for frequency calculation, set to 'geometric'.
- ``ffn_act``: Activation function used in the MLP layers.
- ``ffn_num_hidden_layers``: Number of layers in the feed-forward network.
- ``ffn_dropout_rate``: Dropout rate for regularization within the MLP.
- ``ffn_hidden_dim``: Dimension of each hidden layer within the MLP.
- ``ffn_use_layernormalize``: Boolean to enable normalization within the MLP.
- ``ffn_skip_connection``: Enables skip connections within the MLP, potentially enhancing learning.
- ``ffn_context_str``: Context string for debugging and detailed logging within the network.

Methods
--------

.. method:: forward(coords)
    :no-index:

Processes input coordinates through the location encoder to generate final spatial embeddings.
- **Parameters**:
    - `coords` (List or np.ndarray): Coordinates to process, formatted as `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
    - `sprenc` (Tensor): Spatial relation embeddings with a shape of `(batch_size, num_context_pt, spa_embed_dim)`.

:class:`SphereMixScaleSpatialRelationPositionEncoder`
=====================================================

Overview
--------

Transforms spatial coordinates into high-dimensional encoded formats using sinusoidal functions scaled across multiple frequencies, enhancing the model's capability to discern spatial nuances.

Assumptions for Grid-Structured Data
-------------------------------------

Spatial Regularity
~~~~~~~~~~~~~~~~~~

Grid data often comes in regular, evenly spaced intervals, such as pixels in images or cells in raster GIS data.

Two-Dimensional Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Most grid data is two-dimensional, requiring simultaneous encoding of both dimensions to capture spatial relationships effectively.

Formula Development
~~~~~~~~~~~~~~~~~~~

Base Sinusoidal Encoding
+++++++++++++++++++++++++

For each coordinate component $x$ and $y$, apply sinusoidal functions across multiple scales:

:math:`E(x, y) = \bigoplus{}^{L-1}_{i=0} \left\[ \sin(\omega_i x), \cos(\omega_i x), \sin(\omega_i y), \cos(\omega_i y) \right\]`

Where:
- :math:`\bigoplus` denotes vector concatenation.
- :math:`L` is the number of different frequencies used.
- :math:`\omega_i` are the scaled frequencies.

Frequency Scaling
+++++++++++++++++

Given the grid structure, frequency scaling might be adapted based on typical distances or resolutions encountered in grid data:

:math:`\omega_i = \pi \cdot \left(\frac{2^i}{\text{cell size}}\right)`

This scaling method aligns the frequency increments with the spatial resolution of grid cells, allowing the encoder to capture variations within and between cells.

Enhanced Spatial Encoding
+++++++++++++++++++++++++

To account for the two-dimensional nature of grid data and potentially the interactions between grid cells, the encoding can be expanded to include mixed terms that combine :math:`x` and :math:`y` coordinates:

:math:`E_{\text{enhanced}}(x, y) = E(x, y) \oplus \left\[\sin(\omega_i x) \cdot \cos(\omega_i y), \cos(\omega_i x) \cdot \sin(\omega_i y)\right\]`

These mixed terms help to model cross-dimensional spatial interactions, which are critical in grid-like structures where horizontal and vertical relationships might influence the spatial analysis.

Output Dimensionality
++++++++++++++++++++++

The output dimensionality, considering the enhanced encoding, becomes:

:math:`\text{Output Dim} = 4L + 2L = 6L`

Where :math:`4L` comes from the original sinusoidal terms for :math:`x` and :math:`y`, and :math:`2L` from the mixed terms added for cross-dimensional interactions.

Features
~~~~~~~~

- **Geometric Frequency Scaling**: Employs a geometric progression of frequencies for sinusoidal encoding, capturing a broad range of spatial details.
- **Configurable Parameters**: Supports adjustments in encoding dimensions, frequency range, and computational resources.

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

- ``coord_dim``: The dimensionality of the space being encoded.
- ``frequency_num``: The number of frequencies used for encoding.
- ``device``: Specifies the computational device.

Methods
~~~~~~~

.. method:: cal_elementwise_angle(coord, cur_freq)
    :no-index:
- **Parameters**:
    - `coord`: The deltaX or deltaY.
    - `cur_freq`: The frequency index.
- **Returns**:
    - The calculated angle for the sinusoidal transformation.

.. method:: cal_coord_embed(coords_tuple)
    :no-index:

Converts a batch of coordinates into sinusoidally-encoded vectors.
- **Parameters**:
    - `coords_tuple`: Tuple of spatial differences.
- **Returns**:
    - High-dimensional vector representing the encoded spatial relationships.

Usage Example
-------------

.. code-block:: python

    encoder = SphereMixScaleSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    coord_dim=2,
    device="cuda",
    frequency_num=16,
    max_radius=10000,
    min_radius=10,
    freq_init="geometric",
    ffn_act="relu",
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_hidden_dim=256,
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    ffn_context_str="SphereMixScaleSpatialRelationEncoder"
    )

    coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
    embeddings = encoder.forward(coords)