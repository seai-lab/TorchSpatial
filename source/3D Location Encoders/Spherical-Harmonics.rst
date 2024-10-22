*Siren(SH)*
++++++++++++++++++++++

:class:`SphericalHarmonicsSpatialRelationLocationEncoder`
=========================================================

The :class:`SphericalHarmonicsSpatialRelationLocationEncoder` is designed to encode spatial relationships using spherical harmonics, which are particularly useful for modeling functions on the sphere. This encoder is complemented by the :class:`SphericalHarmonicsSpatialRelationPositionEncoder`, which transforms geographical coordinates into a three-dimensional space and applies spherical harmonics for positional encoding.

Features
--------

- **Position Encoding** ``self.position_encoder``: Utilizes :class:`SphericalHarmonicsSpatialRelationPositionEncoder` for converting longitude and latitude into 3D coordinates and encoding them using spherical harmonics.
- **Feed-Forward Neural Network** ``self.ffn``: Processes the spherical harmonics encoded data through a multi-layer feed-forward neural network to generate final spatial embeddings.

Configuration Parameters
------------------------

- ``spa_embed_dim``: Dimensionality of the output spatial embeddings.
- ``coord_dim``: Dimensionality of the coordinate space, typically 2 for geographical coordinates.
- ``legendre_poly_num``: Number of Legendre polynomials used in the spherical harmonics computation.
- ``device``: Computation device used (e.g., 'cuda' for GPU acceleration).
- ``ffn_act``: Activation function for the neural network layers.
- ``ffn_num_hidden_layers``: Number of hidden layers in the neural network.
- ``ffn_dropout_rate``: Dropout rate to prevent overfitting during training.
- ``ffn_hidden_dim``: Dimension of each hidden layer within the network.
- ``ffn_use_layernormalize``: Whether to use layer normalization.
- ``ffn_skip_connection``: Whether to include skip connections within the network layers.
- ``ffn_context_str``: Context string for debugging and detailed logging within the network.

Methods
-------

.. method:: forward(coords)
    :no-index:

- **Purpose**: Processes input coordinates through the encoder to produce spatial embeddings.
- **Parameters**:
    - ``coords`` (List or np.ndarray): Coordinates to process, formatted as `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
    - ``sprenc`` (Tensor): The final spatial relation embeddings, shaped `(batch_size, num_context_pt, spa_embed_dim)`.

:class:`SphericalHarmonicsSpatialRelationPositionEncoder`
=========================================================

Overview
--------

This position encoder transforms geographic coordinates (longitude and latitude) into a 3D space using spherical coordinates, and then applies spherical harmonics to produce a high-dimensional representation of these positions.

Features
--------

- **3D Coordinate Conversion**: Converts longitude and latitude into 3D spherical coordinates.
- **Spherical Harmonics Encoding**: Applies spherical harmonics to encode the positions in a high-dimensional space, capturing complex spatial relationships.

Formula
-------

The encoder utilizes spherical harmonics to encode spatial data, transforming coordinates (longitude and latitude) into a three-dimensional spherical coordinate system, and then applying spherical harmonics to these coordinates.

**1. Conversion to Spherical Coordinates**

Given longitude ( :math:`\phi`  ) and latitude ( :math:`\theta`  ), the coordinates are converted into spherical coordinates. Each point on the surface of the sphere is expressed as:

.. math::
    x = \cos(\phi) \sin(\theta)
.. math::
    y = \sin(\phi) \sin(\theta)
.. math::
    z = \cos(\theta)

Where:

    -  :math:`\phi` is longitude in radians.
    -  :math:`\theta` is latitude in radians.

**2. Spherical Harmonics**

Spherical harmonics are orthogonal functions defined on the sphere, used to generate a positional encoding. The function :math:`Y_l^m(\theta, \phi)` for a degree :math:`l` and order :math:`m` is given by:

.. math::
    Y_l^m(\theta, \phi) = P_l^m(\cos(\theta)) e^{im\phi}

Where:

    - :math:`P_l^m` are the associated Legendre polynomials.
    - :math:`e^{im\phi}` is the complex exponential function.

**3. Encoding Formula**

The position encoding using spherical harmonics is computed as a sum of these functions across a range of degrees and orders, generally formulated as:

.. math::
    \text{Enc}(x, y, z) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)

Where:

    - :math:`c_{lm}` are coefficients, which may be learned or predefined.
    - :math:`L` is the maximum degree of spherical harmonics used, determined by the `legendre_poly_num`.

These embeddings are then processed through a feed-forward neural network, incorporating linear transformations and non-linear activations to produce the final spatial relation embeddings suitable for machine learning applications.

Configuration Parameters
------------------------

- ``coord_dim``: Dimensionality of the input space, typically 2 for (longitude, latitude).
- ``legendre_poly_num``: Number of Legendre polynomials used for spherical harmonics.
- ``device``: Specifies the computation device (e.g., 'cuda').

Methods
-------

.. method:: make_output_embeds(coords)
    :no-index:

- **Description**: Converts geographical coordinates into embeddings using spherical harmonics.
- **Parameters**:
    - ``coords``: Coordinates in the format `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
    - High-dimensional embeddings representing the input data in terms of spherical harmonics.

.. method:: forward(coords)
    :no-index:

- **Description**: Encodes a list of geographic coordinates into their spherical harmonics embeddings.
- **Parameters**:
    - ``coords``: A list of coordinates.
- **Returns**:
    - Tensor of spatial relation embeddings shaped as `(batch_size, num_context_pt, pos_enc_output_dim)`.   

Usage Example
=============

.. code-block:: python

    # Initialize the encoder
    encoder = SphericalHarmonicsSpatialRelationLocationEncoder(
        spa_embed_dim=64,
        coord_dim=2,
        legendre_poly_num=8,
        device="cuda",
        ffn_act="relu",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_hidden_dim=256,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_context_str="SphericalHarmonicsSpatialRelationEncoder"
    )

    # Example coordinate data
    coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])
    embeddings = encoder.forward(coords)

