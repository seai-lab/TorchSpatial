# <em>Space2Vec-grid</em>

> # GridCellSpatialRelationLocationEncoder

## Overview
The `GridCellSpatialRelationLocationEncoder` is designed for encoding spatial relations between locations. This encoder integrates a position encoding strategy, leveraging a `GridCellSpatialRelationPositionEncoder`, and further processes the encoded positions through a customizable multi-layer feed-forward neural network.

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

> ## GridCellSpatialRelationPositionEncoder

### Features
- **Sinusoidal Encoding**: Utilizes sinusoidal functions to encode spatial differences, allowing for the representation of these differences in a form that neural networks can more effectively learn from.

### Configuration Parameters
- **coord_dim**: Dimensionality of the space being encoded (e.g., 2D, 3D).
- **frequency_num**: Number of different sinusoidal frequencies used to encode spatial differences.
- **max_radius**: Maximum spatial context radius, defining the upper scale of encoding.
- **min_radius**: Minimum spatial context radius, defining the lower scale of encoding.
- **freq_init**: Method to initialize the frequency list, can be 'random', 'geometric', or 'nerf'.
- **device**: Specifies the computational device, e.g., 'cuda' for GPU acceleration.

### Methods

#### `cal_elementwise_angle(coord, cur_freq)`
Calculates the angle for sinusoidal function based on the coordinate difference and current frequency.
- **Parameters**:
  - `coord`: Spatial difference (deltaX or deltaY).
  - `cur_freq`: Current frequency index.
- **Returns**:
  - Calculated angle for sinusoidal transformation.

#### `cal_coord_embed(coords_tuple)`
Converts a tuple of coordinates into an embedded format using sinusoidal encoding.
- **Parameters**:
  - `coords_tuple`: Tuple containing deltaX and deltaY.
- **Returns**:
  - High-dimensional vector representing the embedded coordinates.

#### `cal_pos_enc_output_dim()`
Calculates the output dimension of the position-encoded spatial relationship.
- **Returns**:
  - The dimension of the encoded spatial relation embedding.

#### `cal_freq_list()`
Calculates the list of frequencies used for the sinusoidal encoding based on the initialization method specified.
- **Modifies**:
  - Internal frequency list based on the maximum and minimum radii and the total number of frequencies.

#### `cal_freq_mat()`
Generates a matrix of frequencies to be used for batch processing of spatial data.
- **Modifies**:
  - Internal frequency matrix to match the dimensions required for vectorized operations.

#### `make_output_embeds(coords)`
Processes a batch of coordinates and converts them into spatial relation embeddings.
- **Parameters**:
  - `coords`: Batch of spatial differences.
- **Returns**:
  - Batch of spatial relation embeddings in high-dimensional space.
> 
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
