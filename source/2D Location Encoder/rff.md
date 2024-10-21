# <em>rff</em>

> # RFFSpatialRelationLocationEncoder

## Overview
The `RFFSpatialRelationLocationEncoder` is designed to process spatial relations between locations using Random Fourier Features (RFF) adapted for spatial encoding. It utilizes the `RFFSpatialRelationPositionEncoder` to transform spatial coordinates into a high-dimensional space, enhancing the model's ability to capture and interpret spatial relationships across various scales.

## Features
- **Position Encoding (`self.position_encoder`)**: Utilizes `RFFSpatialRelationPositionEncoder` for transforming spatial differences into frequency-based representations using Random Fourier Features.
- **Feed-Forward Neural Network (`self.ffn`)**: Processes the RFF-based data through a multi-layer neural network to generate final spatial embeddings.

## Configuration Parameters
- `spa_embed_dim`: Dimensionality of the output spatial embeddings.
- `coord_dim`: Dimensionality of the coordinate space (typically 2D).
- `frequency_num`: Number of frequency components used in the positional encoding.
- `rbf_kernel_size`: Size of the RBF kernel used in the generation of direction vectors.
- `extent`: The extent of the coordinate space (optional).
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

## RFFSpatialRelationPositionEncoder

### Overview
The `RFFSpatialRelationPositionEncoder` leverages Random Fourier Features (RFF) to encode spatial coordinates into high-dimensional representations. This method is based on the paper "Random Features for Large-Scale Kernel Machines" and is particularly effective for approximating kernel functions.

### Features
- **Random Fourier Feature Encoding**: Transforms spatial data into a frequency-based representation, capturing inherent spatial frequencies and patterns effectively.
- **Adaptable to Different Spatial Extents**: Can normalize input coordinates based on the provided spatial extent.

## Theory

### Random Fourier Feature (RFF) Encoding

Random Fourier Features provide an approximation to shift-invariant kernel functions by mapping the input data into a randomized low-dimensional feature space. The key idea is to use random projections to approximate the kernel function.

### Gaussian RBF Kernel Approximation

The Gaussian RBF kernel is defined as:
$K(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$
where $\|x - y\|$ is the Euclidean distance between points $x$ and $y$, and$\sigma$ is the kernel size (bandwidth).

### Random Fourier Features

Using Bochner's theorem, any shift-invariant kernel can be represented as the Fourier transform of a probability measure. For the Gaussian RBF kernel, the transformation is given by:
$z(x) = \sqrt{\frac{2}{D}} \cos(\omega^T x + b)$
where$ \omega$ is drawn from a Gaussian distribution, $b$ is drawn from a uniform distribution, and $D$ is the dimension of the feature space.

### Formulas

1. **Generate Direction and Shift Vectors**:
   - Direction vector $\omega$:
     $\omega \sim \mathcal{N}(0, \sigma^2 I)$
   - Shift vector $b$:
     
    $b \sim \text{Uniform}(0, 2\pi)$

2. **Random Fourier Feature Transformation**:
   
   $z(x) = \sqrt{\frac{2}{D}} \cos(\omega^T x + b)$

### Implementation

#### `generate_direction_vector()`
- **Purpose**: Generates the direction (omega) and shift (b) vectors used in the RFF transformation.
- **Returns**:
  - `dirvec`: Direction vectors.
  - `shift`: Shift vectors.

#### `make_output_embeds(coords)`
- **Purpose**: Converts input coordinates into RFF-based high-dimensional embeddings.
- **Parameters**:
  - `coords`: Input coordinates.
- **Returns**:
  - High-dimensional embeddings representing the input data in the RFF feature space.

## Usage Example
```python
# Initialize the encoder
encoder = RFFSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    coord_dim=2,
    frequency_num=16,
    rbf_kernel_size=1.0,
    extent=None,
    device="cuda",
    ffn_act="relu",
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_hidden_dim=256,
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    ffn_context_str="RFFSpatialRelationEncoder"
)

coords = np.array([[34.0522, -118.2437], [40.7128, -74.0060]])  # Example coordinate data
embeddings = encoder.forward(coords)
