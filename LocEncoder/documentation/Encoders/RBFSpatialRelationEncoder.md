# 1.2.2 RBFSpatialRelationEncoder
## Purpose
`RBFSpatialRelationEncoder` is designed for encoding spatial relations between points using Radial Basis Function (RBF) kernels. It is capable of computing spatial embeddings either in global position contexts or relative to other spatial positions, making it suitable for tasks such as global position encoding and spatial context position encoding.

## Properties
- `spa_embed_dim`: Specifies the dimensionality of the output spatial embeddings.
- `coord_dim`: Dimensionality of the space being encoded (e.g., 2D, 3D).
- `num_rbf_anchor_pts`: Number of RBF anchor points used in the encoding.
- `rbf_kernal_size`: Defines the size of the RBF kernel, influencing the scale of influence each anchor point has.
- `rbf_kernal_size_ratio`: Adjusts the RBF kernel size based on the distance of an anchor point from the origin, used in relative models.
- `max_radius`: Maximum radius considered for spatial contexts in relative models.
- `device`: Specifies the computation device (e.g., 'cuda' for GPU).

## Functions

### `cal_rbf_anchor_coord_mat()`
- **Description**: Calculates the coordinate matrix for RBF anchor points based on the model type—global or relative.
- **Parameters**: None explicitly required as inputs. Utilizes internal properties such as `train_locs`, `num_rbf_anchor_pts`, `rbf_anchor_pt_ids`, `max_radius`, and `rbf_kernal_size_ratio` to compute the matrix.
- **Returns**: None. Modifies the internal state by setting the `rbf_coords_mat` and optionally `rbf_kernal_size_mat`.

### `make_input_embeds(coords)`
- **Description**: Converts a set of input coordinates into spatial relation embeddings by calculating the squared Euclidean distance to each RBF anchor point and applying the RBF kernel function.
- **Parameters**:
  - `coords` (`List` or `np.ndarray`): Input coordinates with a shape of (batch_size, num_context_pt=1, coord_dim) representing the spatial differences (deltaX, deltaY) between context points.
- **Returns**:
  - `spr_embeds` (`np.ndarray`): The spatial relation embeddings with a shape of (batch_size, num_context_pt, input_embed_dim), derived from the RBF kernel transformations of the distances.

### `forward(coords)`
- **Description**: Feeds the processed coordinates through the encoding mechanism to produce final spatial embeddings, possibly passing through a feed-forward network if defined.
- **Parameters**:
  - `coords` (`List` or `np.ndarray`): Coordinates to be processed, similar to `make_input_embeds`.
- **Returns**:
  - Depending on the setup, either raw RBF outputs or further processed outputs from a feed-forward network are returned. The return shape is generally (batch_size, num_context_pt, spa_embed_dim) if a feed-forward network (`ffn`) is applied.
