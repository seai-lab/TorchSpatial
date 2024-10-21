# <em>rbf</em>

> # RBFSpatialRelationLocationEncoder

## Overview
The `RBFSpatialRelationLocationEncoder` is designed to process spatial relations between locations using Radial Basis Function (RBF) principles adapted for spatial encoding. It utilizes the `RBFSpatialRelationPositionEncoder` to transform spatial coordinates into a high-dimensional space, enhancing the model's ability to capture and interpret spatial relationships across various scales.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes `RBFSpatialRelationPositionEncoder` for transforming spatial differences into RBF-based representations.
- **Feed-Forward Neural Network (`self.ffn`)**: Processes the RBF-based data through a multi-layer neural network to generate final spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the output spatial embeddings.
- `train_locs`: Training locations used to sample RBF anchor points.
- `model_type`: Type of the model, either 'global' or 'relative'.
- `coord_dim`: Dimensionality of the coordinate space (typically 2D).
- `device`: Computation device used (e.g., 'cuda' for GPU acceleration).
- `num_rbf_anchor_pts`: Number of RBF anchor points.
- `rbf_kernel_size`: Size of the RBF kernel.
- `rbf_kernel_size_ratio`: Ratio used to adjust the kernel size based on the distance from the origin (applied in relative models).
- `max_radius`: Maximum distance considered for spatial interactions.
- `rbf_anchor_pt_ids`: IDs of the RBF anchor points.
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

> ## RBFSpatialRelationPositionEncoder

### Overview
This position encoder leverages Radial Basis Function (RBF) techniques to encode spatial coordinates, enabling the model to recognize patterns and relationships that are not immediately apparent in the spatial domain.
### Theory

#### Radial Basis Function (RBF) Encoding

An RBF is a real-valued function whose value depends only on the distance from a center point, called an anchor point.
The RBF commonly used is the Gaussian function, which measures the similarity between data points based on their Euclidean distance.

#### Gaussian RBF Kernel

The Gaussian RBF kernel is defined as:

$K(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$

where $\|x - y\|$ is the Euclidean distance between points $x$ and $y$, and $\sigma$ is the kernel size (also called the bandwidth).

#### Adaptive Kernel Sizes

In some models, the kernel size $\sigma$ can vary based on the distance from the origin or another reference point.

### Formulas

#### Distance Calculation

For each coordinate $(x, y)$ and each RBF anchor point $(a_i, b_i)$, the Euclidean distance is calculated as:

$d_i = \sqrt{(x-a_i)^2 + (y-b_i)^2}$

#### Gaussian RBF Encoding

The RBF encoding for each distance $d_i$ with kernel size $\sigma_i$ is calculated as:

$\text{RBF}_i = \exp\left(-\frac{d_i^2}{2\sigma_i^2}\right)$

If a kernel size ratio is applied, $\sigma_i$ may be adjusted based on the distance from the origin:

$\sigma_i = d_i \times \text{rbf-kernel-size-ratio} + \text{rbf-kernel-size}$

### Features
- **RBF Encoding**: Transforms spatial data into an RBF-based representation, capturing inherent spatial patterns effectively.
- **Adaptive Kernel Sizes**: Allows the kernel sizes to adapt based on the distance from the origin in relative models.

### Configuration Parameters
- `model_type`: Type of the model, either 'global' or 'relative'.
- `train_locs`: Training locations used to sample RBF anchor points.
- `coord_dim`: Dimensionality of the space being encoded.
- `num_rbf_anchor_pts`: Number of different RBF anchor points used in the encoding.
- `rbf_kernel_size`: The RBF kernel size.
- `rbf_kernel_size_ratio`: Ratio used to adjust the kernel size based on the distance from the origin (applied in relative models).
- `max_radius`: The maximum effective radius for the encoding.
- `rbf_anchor_pt_ids`: IDs of the RBF anchor points.
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
encoder = RBFSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    train_locs=np.array([[34.0522, -118.2437], [40.7128, -74.0060]]),  # Example train_locs
    model_type='global',
    coord_dim=2,
    device="cuda",
    num_rbf_anchor_pts=100,
    rbf_kernel_size=10e2,
    rbf_kernel_size_ratio=0.0,
    max_radius=10000,
    ffn_act="relu",
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_hidden_dim=256,
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    ffn_context_str="RBFSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
