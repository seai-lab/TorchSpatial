# SphereGridSpatialRelationLocationEncoder

## Overview
The `SphereGridSpatialRelationLocationEncoder` is engineered for encoding spatial relationships between locations. It leverages the `SphereGridSpatialRelationPositionEncoder` to initially encode spatial differences, then processes these through a customizable multi-layer feed-forward neural network to produce high-dimensional spatial embeddings.

## Features
- **Position Encoding**: Uses the `SphereGridSpatialRelationPositionEncoder` for encoding spatial differences using sinusoidal functions.
- **Feed-Forward Neural Network**: Converts the position-encoded data into spatial embeddings through multiple neural network layers.

## Configuration Parameters
- **spa_embed_dim**: The dimensionality of the spatial embedding output.
- **coord_dim**: The dimensionality of the coordinate space (e.g., 2D, 3D).
- **device**: Computation device (e.g., 'cuda').
- **frequency_num**: Number of frequency components used in positional encoding.
- **max_radius**: Maximum spatial context radius.
- **min_radius**: Minimum spatial context radius.
- **freq_init**: Initialization method for frequency calculation, set to 'geometric'.
- **ffn_act**: Activation function for the feed-forward layers.
- **ffn_num_hidden_layers**: Number of hidden layers in the feed-forward network.
- **ffn_dropout_rate**: Dropout rate used in the feed-forward network.
- **ffn_hidden_dim**: Dimension of each hidden layer in the feed-forward network.
- **ffn_use_layernormalize**: Flag to enable layer normalization in the network.
- **ffn_skip_connection**: Flag to enable skip connections in the network.
- **ffn_context_str**: Context string for debugging and detailed logging.

## Methods
### `forward(coords)`
- **Purpose**: Processes input coordinates through the encoder to produce final spatial embeddings.
- **Parameters**:
  - `coords` (List or np.ndarray): Coordinates to be processed, expected in the format `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
  - `sprenc` (Tensor): Spatial relation embeddings, shaped `(batch_size, num_context_pt, spa_embed_dim)`.

> ## SphereGridSpatialRelationPositionEncoder

### Features
- **Sinusoidal Encoding**: Applies sinusoidal functions to encode spatial differences, enhancing the model's ability to learn from these features.
- **Configurable Parameters**: Supports customization of encoding parameters such as space dimensionality and computation device.
    <p align="center">
      <img src="../figs/sphereC+.png" alt="sphereC-plus-transformation" title="sphereC-plus-transformation" width="80%" />
    </p>
### Configuration Parameters
- **coord_dim**: Dimensionality of the space being encoded (e.g., 2D, 3D).
- **frequency_num**: Number of frequencies used in sinusoidal encoding.
- **device**: Specifies the computational device.

### Methods
#### `make_output_embeds(coords)`
- **Description**: Converts a batch of coordinates into spatial relation embeddings.
- **Parameters**:
  - `coords`: Spatial differences to be encoded.
- **Returns**:
  - Spatial relation embeddings in high-dimensional space.

#### `forward(coords)`
- **Description**: Feeds processed coordinates through the encoder to generate final spatial embeddings.
- **Parameters**:
  - `coords`: Coordinates to process.
- **Returns**:
  - Tensor of spatial relation embeddings.

## Usage Example
```python
encoder = SphereGridSpatialRelationLocationEncoder(
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
    ffn_context_str="SphereGridSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
