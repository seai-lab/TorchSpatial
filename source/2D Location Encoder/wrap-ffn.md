# <em>wrap-ffn</em>

> # AodhaFFNSpatialRelationLocationEncoder

## Overview
The `AodhaFFNSpatialRelationLocationEncoder` is designed to process spatial relations between locations using Fourier Feature Transform (FFT) adapted for spatial encoding. It utilizes the `AodhaFFTSpatialRelationPositionEncoder` to transform spatial coordinates into a high-dimensional space, enhancing the model's ability to capture and interpret spatial relationships across various scales.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes `AodhaFFTSpatialRelationPositionEncoder` for transforming spatial differences into frequency-based representations using Fourier Feature Transform.
- **Feed-Forward Neural Network (`self.ffn`)**: Processes the FFT-based data through a multi-layer neural network to generate final spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the output spatial embeddings.
- `extent`: The extent of the coordinate space (x_min, x_max, y_min, y_max).
- `coord_dim`: Dimensionality of the coordinate space (typically 2D).
- `do_pos_enc`: Whether to perform position encoding.
- `do_global_pos_enc`: Whether to normalize coordinates globally.
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

## AodhaFFTSpatialRelationPositionEncoder

### Overview
The `AodhaFFTSpatialRelationPositionEncoder` leverages Fourier Feature Transform (FFT) to encode spatial coordinates into high-dimensional representations. This method divides the space into grids and uses grid embeddings to represent the spatial relations.

### Features
- **Fourier Feature Transform Encoding**: Transforms spatial data into a frequency-based representation, capturing inherent spatial frequencies and patterns effectively.
- **Adaptable to Different Spatial Extents**: Can normalize input coordinates based on the provided spatial extent.

## Theory

### Fourier Feature Transform (FFT) Encoding

Fourier Feature Transform provides an approximation to continuous signals by mapping the input data into a randomized low-dimensional feature space using sine and cosine functions.

### Fourier Transform Basis Functions

The Fourier Transform uses sine and cosine functions as basis functions to represent the original signal. In spatial encoding, this can be represented as:
$\text{sin}(\pi \cdot x), \text{cos}(\pi \cdot x), \text{sin}(\pi \cdot y), \text{cos}(\pi \cdot y)$

### Formulas

1. **Normalization**:
   - Normalize coordinates based on the extent of the coordinate space.
     
   $x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$

   $y_{\text{norm}} = \frac{y - y_{\min}}{y_{\max} - y_{\min}}$

2. **Fourier Feature Transform**:
   - Apply sine and cosine functions to the normalized coordinates: $\[\text{sin}(\pi \cdot x_{\text{norm}}), \text{cos}(\pi \cdot x_{\text{norm}}),$ $\text{sin}(\pi \cdot y_{\text{norm}}), \text{cos}(\pi \cdot y_{\text{norm}})\]$

### Implementation

#### `make_output_embeds(coords)`
- **Purpose**: Converts input coordinates into FFT-based high-dimensional embeddings.
- **Parameters**:
  - `coords`: Input coordinates.
- **Returns**:
  - High-dimensional embeddings representing the input data in the FFT feature space.

## Usage Example
```python
# Initialize the encoder
encoder = AodhaFFNSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    extent=(0, 100, 0, 100),  # Example extent
    coord_dim=2,
    do_pos_enc=True,
    do_global_pos_enc=True,
    device="cuda",
    ffn_act="relu",
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_hidden_dim=256,
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    ffn_context_str="AodhaFFTSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
