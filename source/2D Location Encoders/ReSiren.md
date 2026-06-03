# <em>ReSiren</em>
> # ReSirenSpatialRelationLocationEncoder

## Overview

The `ReSirenSpatialRelationLocationEncoder` encodes geographic coordinates with a **SIREN**
(Sinusoidal Representation Network) that uses **residual connections** (ReSiren). It first
rescales the input `(lon, lat)` to `[-1, 1]` and then passes them through a stack of sinusoidal
layers to produce a spatial embedding.

A plain SIREN is hard to deepen: every layer expects a specific pre-activation distribution, so
naive residual connections on the post-activation outputs break that constraint. ReSiren instead
adds the residual on the **pre-activation hidden vectors** (the "gaussian" before the sine),
which are well-behaved Gaussians. This keeps the residual stream inside SIREN's distributional
constraints while letting the network scale to many more layers, which improves performance
across a variety of ecological and non-ecological tasks.

ReSiren was introduced in the Climplicit project:

> Dollinger et al., *Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks*,
> ICLR 2025 Workshops. [arXiv:2504.05089](https://arxiv.org/abs/2504.05089)

The SIREN base is adapted from the [SatCLIP location encoder](https://github.com/microsoft/satclip/blob/main/satclip/location_encoder.py).

## Features

- **Position Encoding** (`self.position_encoder`): a `ReSirenSpatialRelationPositionEncoder`
  that linearly rescales `(lon, lat)` from degrees to `[-1, 1]` (the input range SIREN expects).
- **Residual SIREN network** (`self.siren`): a `SirenNet` of sinusoidal layers with residual
  connections on the pre-activation hidden vectors. The first layer additionally uses the
  H-SIREN `sinh` activation.

## Configuration Parameters

- `spa_embed_dim`: dimensionality of the output spatial embedding.
- `coord_dim`: dimensionality of the coordinate space (2 for lon/lat).
- `device`: computation device (e.g., `'cuda'` or `'cpu'`).
- `extent`: `(lon_min, lon_max, lat_min, lat_max)` used to rescale coordinates to `[-1, 1]`.
  Defaults to `(-180, 180, -90, 90)`.
- `siren_num_layers`: number of hidden SIREN layers (a final linear SIREN layer is appended).
  Residual connections make it safe to use many layers (e.g. 16).
- `siren_dim_hidden`: hidden dimension of each SIREN layer.
- `siren_w0`: angular frequency of the hidden layers (SIREN's `w0`).
- `siren_w0_initial`: angular frequency of the first layer (SIREN's larger initial frequency).
- `h_siren`: whether to use the H-SIREN `sinh` activation after the first layer.
- `residual_connections`: whether to enable the ReSiren residual connections.
- `ffn_context_str`: context string for debugging/logging.

When used through TorchSpatial's `get_spa_encoder` factory (`--spa_enc_type ReSiren`),
`siren_num_layers` and `siren_dim_hidden` are taken from the `num_hidden_layer` and `hidden_dim`
parameters, and `h_siren` / `residual_connections` are enabled by default.

## Recommended hyperparameters

ReSiren's residual connections exist specifically so the network can scale to **many layers** —
that is where it improves over a plain SIREN. It should therefore **not** be run at the repo
default of `--num_hidden_layer 1`, which collapses it to a shallow SIREN and hides its advantage.

- `--num_hidden_layer` (SIREN depth): sweep over deeper values, e.g. `4, 8, 12, 16`. Climplicit
  (arXiv:2504.05089) used **16**.
- `--hidden_dim` (SIREN width): e.g. `256, 512, 1024` (Climplicit used **512**).
- `--num_filts` (output embedding dim): keep at the shared default **256** for fair comparison.
- ReSiren ignores the positional-encoding knobs the other encoders sweep (`--frequency_num`,
  `--min_radius`, `--max_radius`, `--spa_f_act`): it uses the Direct `[-1, 1]` encoding and a
  fixed sine activation.

A ready-to-run depth sweep is provided in `main/run_sh/run_resiren.sh`.

## Methods

### `forward(coords)`

- **Purpose**: encode a batch of coordinates into spatial embeddings.
- **Parameters**:
  - `coords` (list or `np.ndarray`): coordinates of shape `(batch_size, num_context_pt, coord_dim)`,
    holding `(lon, lat)` in **degrees**.
- **Returns**:
  - `sprenc` (`Tensor`): spatial embeddings of shape `(batch_size, num_context_pt, spa_embed_dim)`.

## Usage Example

```python
import numpy as np
from SpatialRelationEncoder import ReSirenSpatialRelationLocationEncoder

# Initialize the encoder
encoder = ReSirenSpatialRelationLocationEncoder(
    spa_embed_dim=64,
    coord_dim=2,
    device="cpu",
    siren_num_layers=16,
    siren_dim_hidden=512,
    h_siren=True,
    residual_connections=True,
)

# Sample coordinates: (batch_size, num_context_pt, coord_dim), (lon, lat) in degrees
coords = np.array([[[8.55, 47.40]], [[-74.0060, 40.7128]]])

# Generate spatial embeddings -> shape (2, 1, 64)
embeddings = encoder(coords)
print(embeddings.shape)
```
