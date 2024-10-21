# DFTSpatialRelationLocationEncoder

## Overview
The `DFTSpatialRelationLocationEncoder` is designed to process spatial relations between locations using Discrete Fourier Transform (DFT) principles adapted for spatial encoding. It utilizes the `DFTSpatialRelationPositionEncoder` to transform spatial coordinates into a frequency domain, enhancing the model's ability to capture and interpret spatial relationships across various scales.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes `DFTSpatialRelationPositionEncoder` for transforming spatial differences into frequency-based representations.
- **Feed-Forward Neural Network (`self.ffn`)**: Processes the frequency domain data through a multi-layer neural network to generate final spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the output spatial embeddings.
- `coord_dim`: Dimensionality of the coordinate space (typically 2D).
- `frequency_num`: Number of frequency components used in the positional encoding.
- `max_radius`: Maximum distance considered for spatial interactions.
- `min_radius`: Minimum distance that can be resolved by the encoding.
- `freq_init`: Method used for initializing the frequency components ('geometric' suggests a regular scaling).
- `device`: Computation device used (e.g., 'cuda' for GPU acceleration).
- `ffn_act`: Activation function for the neural network layers.
- `ffn_num_hidden_layers`: Number of hidden layers in the neural network.
- `ffn_dropout_rate`: Dropout rate to prevent overfitting during training.
- `ffn_hidden_dim`: Dimension of each hidden layer within the network.
- `ffn_use_layernormalize`: Whether to use layer normalization.
- `ffn_skip_connection`: Whether to include skip connections within the network layers.
- `ffn_context_str`: Context string for debugging and detailed logging within the network.

## Methods
### `forward(coords)`
- **Purpose**: Processes input coordinates through the encoder to produce spatial embeddings.
- **Parameters**:
  - `coords` (List or np.ndarray): Coordinates to process, formatted as `(batch_size, num_context_pt, coord_dim)`.
- **Returns**:
  - `sprenc` (Tensor): The final spatial relation embeddings, shaped `(batch_size, num_context_pt, spa_embed_dim)`.

> ## DFTSpatialRelationPositionEncoder

### Overview
This position encoder leverages Discrete Fourier Transform (DFT) techniques to encode spatial coordinates into the frequency domain, enabling the model to recognize patterns and relationships that are not immediately apparent in the spatial domain.
    <p align="center">
      <img src="../images/dfs.png" alt="dfs-transformation" title="dfs-transformation" width="80%" />
    </p>
### Features
- **Frequency Domain Conversion**: Transforms spatial data into a frequency-based representation, capturing inherent spatial frequencies and patterns effectively.
- **Multi-Scale Analysis**: By varying the number of frequencies and their initialization, the encoder can adapt to different spatial scales and resolutions.

### Configuration Parameters
- `coord_dim`: Dimensionality of the space being encoded.
- `frequency_num`: Number of different frequencies used in the encoding.
- `max_radius`: The maximum effective radius for the encoding, influencing the lowest frequency.
- `min_radius`: The minimum effective radius, influencing the highest frequency.
- `freq_init`: The method for initializing the frequencies, impacting how spatial scales are represented.
- `device`: Specifies the computation device.

### Methods

#### `cal_elementwise_angle(coord, cur_freq)`
- **Description**: Calculates the angle for each frequency based on the spatial coordinate.
- **Parameters**:
  - `coord`: Spatial difference, either deltaX or deltaY.
  - `cur_freq`: Current frequency index.
- **Returns**:
  - Computed angle for the transformation.

#### `cal_coord_embed(coords_tuple)`
- **Description**: Encodes a set of coordinates into their frequency domain representations.
- **Parameters**:
  - `coords_tuple`: A tuple of spatial differences.
- **Returns**:
  - High-dimensional vector representing the frequency domain embeddings.

#### `make_output_embeds(coords)`
- **Description**: Converts input spatial data into a comprehensive set of frequency domain features.
- **Parameters**:
  - `coords`: Spatial coordinates to encode.
- **Returns**:
  - High-dimensional embeddings that represent the input data in the frequency domain.

## Usage Example
```python
# Initialize the encoder
encoder = DFTSpatialRelationLocationEncoder(
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
    ffn_context_str="DFTSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
