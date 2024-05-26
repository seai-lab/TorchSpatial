# Space2Vec-grid (GridCellSpatialRelationLocationEncoder)

## Overview
The `GridCellSpatialRelationLocationEncoder` is a sophisticated neural network module designed for encoding spatial relations between locations. This encoder integrates a position encoding strategy, leveraging a `GridCellSpatialRelationPositionEncoder`, and further processes the encoded positions through a customizable multi-layer feed-forward neural network.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes the `GridCellSpatialRelationPositionEncoder` to encode spatial differences (deltaX, deltaY) based on sinusoidal functions.
- **Feed-Forward Neural Network (`self.ffn`)**: Transforms the position-encoded data through a series of feed-forward layers to produce high-dimensional spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the spatial embedding output.
- `coord_dim`: Dimensionality of the coordinate space (e.g., 2D, 3D).
- `frequency_num`: Number of different frequencies used for the sinusoidal encoding.
- `max_radius`: The maximum context radius the model can handle.
- `min_radius`: The minimum context radius, important for defining the scale of positional encoding.
- `freq_init`: Method of initializing the frequency list ('random', 'geometric', 'nerf').
- `device`: Computation device (e.g., 'cuda' for GPU).
- `ffn_act`: Activation function for the feed-forward layers.
- `ffn_num_hidden_layers`: Number of hidden layers in the feed-forward network.
- `ffn_dropout_rate`: Dropout rate used in the feed-forward network.
- `ffn_hidden_dim`: Dimension of each hidden layer in the feed-forward network.
- `ffn_use_layernormalize`: Flag to enable layer normalization in the feed-forward network.
- `ffn_skip_connection`: Flag to enable skip connections in the feed-forward network.
- `ffn_context_str`: Context string for debugging and detailed logging within the network.

## Methods
### `forward(coords)`
- **Purpose**: Processes input coordinates through the location encoder to produce final spatial embeddings.
- **Parameters**:
  - `coords` (List or np.ndarray): Coordinates to process, expected to be in the form `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
  - `sprenc` (Tensor): Spatial relation embeddings with a shape of `(batch_size, num_context_pt, spa_embed_dim)`.

## Usage Example
```python
encoder = GridCellSpatialRelationLocationEncoder(
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
    ffn_context_str="GridCellSpatialRelationEncoder"
)

coords = np.array([...])  # your coordinate data
embeddings = encoder.forward(coords)
