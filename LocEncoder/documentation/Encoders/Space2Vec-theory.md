# Space2Vec-theory - TheoryGridCellSpatialRelationLocationEncoder

## Overview
The `TheoryGridCellSpatialRelationLocationEncoder` extends the `LocationEncoder` to encode spatial relationships between locations using advanced theoretical methods. This encoder uses a specialized position encoder (`TheoryGridCellSpatialRelationPositionEncoder`) to transform spatial differences into a high-dimensional space, and further processes these embeddings through a custom multi-layer feed-forward neural network.

## Features
- **Position Encoding**: Utilizes the [`TheoryGridCellSpatialRelationPositionEncoder`](#TheoryGridCellSpatialRelationPositionEncoder) for converting spatial differences into encoded positions based on specified frequencies and radii.
- **Feed-Forward Neural Network**: Processes the position-encoded data through a multi-layer neural network, customizable in terms of architecture and activation functions.

## Configuration Parameters
- `spa_embed_dim`: The dimensionality of the output spatial embeddings.
- `coord_dim`: Dimensionality of the coordinate space (e.g., 2D).
- `frequency_num`: The number of different frequencies used for sinusoidal encoding.
- `max_radius`: The largest context radius the model can handle.
- `min_radius`: The smallest context radius, essential for defining the scale of positional encoding.
- `freq_init`: Method for initializing the frequency list ('geometric' by default).
- `device`: Computation device (e.g., 'cuda' for GPU operations).
- `ffn_act`: Activation function used in the feed-forward network.
- `ffn_num_hidden_layers`: Number of hidden layers in the feed-forward network.
- `ffn_dropout_rate`: Dropout rate used in the network.
- `ffn_hidden_dim`: Dimension of each hidden layer in the network.
- `ffn_use_layernormalize`: Boolean flag to enable layer normalization in the network.
- `ffn_skip_connection`: Boolean flag to enable skip connections in the network.
- `ffn_context_str`: A string identifier used for context-specific logging or debugging.

> ## TheoryGridCellSpatialRelationPositionEncoder <a name="TheoryGridCellSpatialRelationPositionEncoder"></a>
### Configuration Parameters
- `coord_dim`: Dimensionality of the space being encoded (e.g., 2D, 3D).
- `frequency_num`: Number of different sinusoidal frequencies used to encode spatial differences.
- `max_radius`: Maximum spatial context radius, defining the upper scale of encoding.
- `min_radius`: Minimum spatial context radius, defining the lower scale of encoding.
- `freq_init`: Method for initializing the frequency list, with options such as 'random', 'geometric', or 'nerf'.
- `device`: Specifies the computational device, e.g., 'cuda' for GPU acceleration.

### Methods

#### `cal_freq_mat()`
- **Description**: Updates the frequency matrix to accommodate multi-dimensional encoding.
- **Modifies**: Extends the frequency matrix to match the dimensions required for advanced vectorized operations across multiple unit vectors.

#### `cal_pos_enc_output_dim()`
- **Description**: Calculates the output dimension of the position-encoded spatial relation embedding, taking into account the additional dimensions introduced by multiple unit vectors.
- **Returns**: The dimension of the encoded spatial relation embedding.

#### `make_output_embeds(coords)`
- **Description**: Processes a batch of coordinates and converts them into spatial relation embeddings using advanced trigonometric transformations.
- **Parameters**:
  - `coords`: Batch of spatial differences (e.g., deltaX, deltaY).
- **Returns**: Batch of spatial relation embeddings in high-dimensional space.


## Usage Example
```python
encoder = TheoryGridCellSpatialRelationLocationEncoder(
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
    ffn_context_str="TheoryGridCellSpatialRelationEncoder"
)

coords = np.array([...])  # your coordinate data
embeddings = encoder.forward(coords)
