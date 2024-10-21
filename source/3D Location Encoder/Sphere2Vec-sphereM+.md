# <em>Space2Vec-sphereM+</em>

> # SphereGridMixScaleSpatialRelationLocationEncoder

## Overview
The `SphereGridMixScaleSpatialRelationLocationEncoder` is engineered for advanced spatial encoding, integrating a position encoder that leverages geometrically scaled sinusoidal functions. It processes these encodings through a multi-layer feed-forward neural network to create detailed spatial embeddings.

## Features
- **Position Encoding (`self.position_encoder`)**: Uses the `SphereGridMixScaleSpatialRelationPositionEncoder` to perform multi-scale sinusoidal encoding of spatial differences.
- **Feed-Forward Neural Network (`self.ffn`)**: Converts the position-encoded data into high-dimensional spatial embeddings through several neural network layers.

## Configuration Parameters
- **spa_embed_dim**: The dimensionality of the spatial embeddings output.
- **coord_dim**: The dimensionality of the coordinate space.
- **frequency_num**: Number of frequency components used in positional encoding.
- **max_radius**: Maximum spatial context radius the encoder can handle.
- **min_radius**: Minimum radius for encoding, affecting the granularity of details captured.
- **freq_init**: Frequency initialization method, set to 'geometric'.
- **device**: Computation device, e.g., 'cuda'.
- **ffn_act**: Activation function used in the neural network layers.
- **ffn_num_hidden_layers**: Number of layers in the feed-forward network.
- **ffn_dropout_rate**: Dropout rate to prevent overfitting.
- **ffn_hidden_dim**: Dimension of each hidden layer in the network.
- **ffn_use_layernormalize**: Flag to enable layer normalization in the network.
- **ffn_skip_connection**: Flag to enable skip connections in the network.
- **ffn_context_str**: Context string for detailed logging and debugging within the network.

## Methods
### `forward(coords)`
Processes input coordinates through the location encoder to produce detailed spatial embeddings.
- **Parameters**:
  - **coords** (List or np.ndarray): Coordinates to process, formatted as `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
  - **sprenc** (Tensor): Spatial relation embeddings, shaped `(batch_size, num_context_pt, spa_embed_dim)`.

> ## SphereGridMixScaleSpatialRelationPositionEncoder

### Overview
This position encoder transforms spatial coordinates using a sophisticated sinusoidal encoding method, featuring multiple scales to capture a wide range of spatial details.
    <p align="center">
      <img src="../images/sphereM+.png" alt="sphereM-plus-transformation" title="sphereM-plus-transformation" width="60%" />
    </p>

### Features
- **Multi-Scale Sinusoidal Encoding**: Applies sinusoidal functions at multiple scales to encode spatial differences, capturing a wide range of spatial details.
- **Geometric Frequency Scaling**: Frequencies increase geometrically, enhancing the encoder's ability to model spatial phenomena at various scales.
### Assumptions
- **Spatial Regularity**: Grid data often comes in regular, evenly spaced intervals, such as pixels in images or cells in raster GIS data.
- **Two-Dimensional Structure**: Most grid data is two-dimensional, requiring simultaneous encoding of both dimensions to capture spatial relationships effectively.

### Configuration Parameters
- **coord_dim**: Dimensionality of the space being encoded.
- **frequency_num**: Total number of different sinusoidal frequencies used.
- **max_radius**: Largest spatial scale considered by the encoder.
- **min_radius**: Smallest spatial scale at which details are captured.
- **freq_init**: Method used to initialize the frequencies, typically 'geometric'.
- **device**: Computation device, such as 'cuda'.

### Methods
#### `cal_elementwise_angle(coord, cur_freq)`
Calculates the angle for sinusoidal encoding based on the coordinate and the current frequency.
- **Parameters**:
  - **coord**: Spatial difference, either deltaX or deltaY.
  - **cur_freq**: Current frequency index.
- **Returns**:
  - Computed angle for the sinusoidal transformation.

#### `cal_coord_embed(coords_tuple)`
Converts a batch of coordinates into sinusoidally-encoded vectors.
- **Parameters**:
  - **coords_tuple**: Tuple of deltaX and deltaY values.
- **Returns**:
  - High-dimensional vector representing the encoded spatial relationships.

#### `cal_output_dim()`
Calculates the dimensionality of the encoded spatial relation embeddings.
- **Returns**:
  - Total dimensionality of the output spatial embeddings.

## Usage Example
```python
encoder = SphereGridMixScaleSpatialRelationLocationEncoder(
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
    ffn_context_str="SphereGridMixScaleSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
