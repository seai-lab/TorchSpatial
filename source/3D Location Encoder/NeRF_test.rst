*NeRF*
======

:class:`NERFSpatialRelationLocationEncoder`
============================================

Overview
--------

The :class:`NERFSpatialRelationLocationEncoder` is designed to compute spatial embeddings from coordinate data using a Neural Radiance Field (NeRF) based encoding approach. This encoder integrates a position encoding strategy, leveraging a :class:`NERFSpatialRelationPositionEncoder`, and further processes the encoded positions through a customizable multi-layer feed-forward neural network.

Features
--------

- **Position Encoding (`self.position_encoder`)**: Utilizes the :class:`NERFSpatialRelationPositionEncoder` to encode spatial differences (latitude, longitude) using NeRF-inspired sinusoidal functions.
- **Feed-Forward Neural Network (`self.ffn`)**: Transforms the position-encoded data through a series of feed-forward layers to produce high-dimensional spatial embeddings.
