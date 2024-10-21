# <em>xyz</em>

> # XYZSpatialRelationLocationEncoder

## Overview
The `XYZSpatialRelationLocationEncoder` is designed for encoding spatial relations between locations. This encoder integrates a position encoding strategy, leveraging an `XYZSpatialRelationPositionEncoder`, and further processes the encoded positions through a customizable multi-layer feed-forward neural network.

## Features
- **Position Encoding (`self.position_encoder`):** Utilizes the `XYZSpatialRelationPositionEncoder` to encode spatial differences (deltaX, deltaY) based on sinusoidal functions.
- **Feed-Forward Neural Network (`self.ffn`):** Transforms the position-encoded data through a series of feed-forward layers to produce high-dimensional spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the spatial embedding output.
- `coord_dim`: Dimensionality of the coordinate space (e.g., 2D, 3D).
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
- **Purpose:** Processes input coordinates through the location encoder to produce final spatial embeddings.
- **Parameters:**
  - `coords` (List or np.ndarray): Coordinates to process, expected to be in the form `(batch_size, num_context_pt, coord_dim)`.
- **Returns:**
  - `sprenc` (Tensor): Spatial relation embeddings with a shape of `(batch_size, num_context_pt, spa_embed_dim)`.

> ## XYZSpatialRelationPositionEncoder

### Features
- **Sinusoidal Encoding:** Utilizes sinusoidal functions to encode spatial differences, allowing for the representation of these differences in a form that neural networks can more effectively learn from.
- **Configurable Parameters:** Allows customization of encoding parameters such as the dimensionality of space and computational device.

### Configuration Parameters
- **coord_dim:** Dimensionality of the space being encoded (e.g., 2D, 3D).
- **device:** Specifies the computational device, e.g., 'cuda' for GPU acceleration.

### Methods

#### `make_output_embeds(coords)`
Processes a batch of coordinates and converts them into spatial relation embeddings.
- **Parameters:**
  - `coords`: Batch of spatial differences.
- **Formulas:**
  - Convert latitude `lat` and longitude `lon` coordinates into radians.
  - Calculate `x, y, z` coordinates using the following equations:

      $$x = \cos(lat) \times \cos(lon)$$

      $$y = \cos(lat) \times \sin(lon)$$  
      
      $$z = \sin(lat)$$
    </p>

      <div style="border:1px solid #000; padding:10px; border-radius:5px;">
      <strong>Note:</strong> If the equations cannot render properly, please click this <a href="https://drive.google.com/uc?id=1tviSk-NbxB0G8fTx5vYpkvBJXmiBK08g">Link</a>. 
      </div>


    Where:

    - *lat* is the latitude coordinate in radians.
    - *lon* is the longitude coordinate in radians.
    - *x*, *y*, *z* are the resulting Cartesian coordinates.

  - Concatenate `x, y, z` coordinates to form the high-dimensional vector representation.

- **Returns:**
  - Batch of spatial relation embeddings in high-dimensional space.

#### `forward(coords)`
Feeds the processed coordinates through the encoder to produce final spatial embeddings.
- **Parameters:**
  - `coords`: Coordinates to process.
- **Returns:**
  - Tensor of spatial relation embeddings.

## Usage Example
```python
encoder = XYZSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    coord_dim=2,
    device="cuda",
    ffn_act="relu",
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_hidden_dim=256,
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    ffn_context_str="XYZSpatialRelationEncoder"
)

coords = np.array([...])  # your coordinate data
embeddings = encoder.forward(coords)
