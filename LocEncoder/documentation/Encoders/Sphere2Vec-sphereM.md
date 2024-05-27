# SphereMixScaleSpatialRelationLocationEncoder Documentation

## Overview
The `SphereMixScaleSpatialRelationLocationEncoder` is engineered to encode spatial relationships between locations using advanced position encoding techniques. It integrates the `SphereMixScaleSpatialRelationPositionEncoder` for initial encoding and processes the results through a multi-layer feed-forward neural network to produce high-dimensional spatial embeddings.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes the `SphereMixScaleSpatialRelationPositionEncoder` to encode spatial differences (deltaX, deltaY) using geometrically scaled sinusoidal functions.
- **Feed-Forward Neural Network (`self.ffn`)**: Transforms position-encoded data through several neural network layers to produce high-dimensional spatial embeddings.

## Configuration Parameters
- **spa_embed_dim**: The dimensionality of the output spatial embeddings.
- **coord_dim**: The dimensionality of the coordinate space, typically 2D.
- **device**: Specifies the computation device, e.g., 'cuda'.
- **frequency_num**: Number of frequency components used in positional encoding.
- **max_radius**: The largest spatial context radius the model can handle.
- **min_radius**: The minimum radius, ensuring detailed capture at smaller scales.
- **freq_init**: Initialization method for frequency calculation, set to 'geometric'.
- **ffn_act**: Activation function used in the MLP layers.
- **ffn_num_hidden_layers**: Number of layers in the feed-forward network.
- **ffn_dropout_rate**: Dropout rate for regularization within the MLP.
- **ffn_hidden_dim**: Dimension of each hidden layer within the MLP.
- **ffn_use_layernormalize**: Boolean to enable normalization within the MLP.
- **ffn_skip_connection**: Enables skip connections within the MLP, potentially enhancing learning.
- **ffn_context_str**: Context string for debugging and detailed logging within the network.

## Methods
### `forward(coords)`
Processes input coordinates through the location encoder to generate final spatial embeddings.
- **Parameters**:
  - `coords` (List or np.ndarray): Coordinates to process, formatted as `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
  - `sprenc` (Tensor): Spatial relation embeddings with a shape of `(batch_size, num_context_pt, spa_embed_dim)`.

> ##  SphereMixScaleSpatialRelationPositionEncoder

### Overview
Transforms spatial coordinates into high-dimensional encoded formats using sinusoidal functions scaled across multiple frequencies, enhancing the model's capability to discern spatial nuances.
    <p align="center">
      <img src="../figs/sphereM.png" alt="sphereM-transformation" title="sphereM-transformation" width="80%" />
    </p>
### Features
- **Geometric Frequency Scaling**: Employs a geometric progression of frequencies for sinusoidal encoding, capturing a broad range of spatial details.
- **Configurable Parameters**: Supports adjustments in encoding dimensions, frequency range, and computational resources.

### Configuration Parameters
- **coord_dim**: The dimensionality of the space being encoded.
- **frequency_num**: The number of frequencies used for encoding.
- **device**: Specifies the computational device.

### Methods
#### `cal_elementwise_angle(coord, cur_freq)`
Calculates the angle for sinusoidal encoding based on the coordinate and the current frequency.
- **Parameters**:
  - `coord`: The deltaX or deltaY.
  - `cur_freq`: The frequency index.
- **Returns**:
  - The calculated angle for the sinusoidal transformation.

#### `cal_coord_embed(coords_tuple)`
Converts a batch of coordinates into sinusoidally-encoded vectors.
- **Parameters**:
  - `coords_tuple`: Tuple of spatial differences.
- **Returns**:
  - High-dimensional vector representing the encoded spatial relationships.

## Usage Example
```python
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
