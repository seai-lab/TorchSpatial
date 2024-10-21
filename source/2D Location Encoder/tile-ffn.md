# <em>tile</em>

> # GridLookupSpatialRelationLocationEncoder

## Overview
The `GridLookupSpatialRelationLocationEncoder` is designed to process spatial relations between locations using a grid-based lookup approach for spatial encoding. It utilizes the `GridLookupSpatialRelationPositionEncoder` to transform spatial coordinates into a high-dimensional space, enhancing the model's ability to capture and interpret spatial relationships across various scales.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes `GridLookupSpatialRelationPositionEncoder` for transforming spatial differences into grid-based representations.
- **Feed-Forward Neural Network (`self.ffn`)**: Processes the grid-based data through a multi-layer neural network to generate final spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the output spatial embeddings.
- `extent`: The extent of the coordinate space (x_min, x_max, y_min, y_max).
- `interval`: The cell size in X and Y direction.
- `coord_dim`: Dimensionality of the coordinate space (typically 2D).
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

> ## GridLookupSpatialRelationPositionEncoder

### Overview
The `GridLookupSpatialRelationPositionEncoder` divides the space into grids and assigns each point to the grid embedding it falls into. This method enables the model to use a grid-based representation for spatial encoding.

### Features
- **Grid-Based Encoding**: Transforms spatial data into a grid-based representation, capturing spatial patterns effectively.
- **Adaptable to Different Spatial Extents**: Can normalize input coordinates based on the provided spatial extent.

## Theory

### Grid-Based Encoding

Grid-based encoding divides the spatial extent into equal-sized cells (grids) and assigns each coordinate to a specific grid cell. Each grid cell has an embedding that represents the spatial location.

### Formulas

1. **Grid Cell Calculation**:
   - Calculate the column and row indices for each coordinate based on the grid interval and extent.
  
  $\text{col} = \left\lfloor \frac{x - x_{\text{min}}}{\text{interval}} \right\rfloor$

  $\text{row} = \left\lfloor \frac{y - y_{\text{min}}}{\text{interval}} \right\rfloor$

2. **Grid Cell Index**:
   - Calculate the unique index for each grid cell.
$\text{index} = \text{row} \times \text{num-cols} + \text{col}$

### Implementation

#### `make_grid_embedding(interval, extent)`
- **Purpose**: Creates grid embeddings for the specified interval and extent.
- **Parameters**:
  - `interval`: The cell size in X and Y direction.
  - `extent`: The extent of the coordinate space.
- **Returns**:
  - Grid embeddings for the specified interval and extent.

#### `make_output_embeds(coords)`
- **Purpose**: Converts input coordinates into grid-based high-dimensional embeddings.
- **Parameters**:
  - `coords`: Input coordinates.
- **Returns**:
  - High-dimensional embeddings representing the input data in the grid-based feature space.

## Usage Example
```python
# Initialize the encoder
encoder = GridLookupSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    extent=(-180, 180, -90, 90),  # Example extent
    interval=1000000,  # Example interval
    coord_dim=2,
    device="cuda",
    ffn_act="relu",
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_hidden_dim=256,
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    ffn_context_str="GridLookupSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
