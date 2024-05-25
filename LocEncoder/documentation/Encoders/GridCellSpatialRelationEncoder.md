# 1.2.1 GridCellSpatialRelationEncoder
## Purpose
The `GridCellSpatialRelationEncoder` is designed to encode spatial relationships between grid cells using a sinusoidal encoding scheme. It is particularly useful in scenarios where precise spatial relations need to be captured, such as navigation systems, robotic mapping, and geographical information systems.

## Properties
- `spa_embed_dim`: Specifies the dimensionality of the output spatial embeddings. This defines the size of the vector that will represent each spatial relationship.
- `coord_dim`: Indicates the dimensionality of the space being encoded, such as 2D for planar surfaces or 3D for volumetric spaces.
- `frequency_num`: The number of different sinusoidal functions used to encode the spatial relationships. A higher number of frequencies can capture more detailed spatial nuances.
- `max_radius`: Defines the maximum spatial context radius that the encoder can handle. This parameter sets the scale of the spatial area within which the relationships are considered relevant.

## Functions
### `cal_elementwise_angle(self, coord, cur_freq)`
#### Description
This function calculates the angle for each element of the input coordinates, scaled by the corresponding frequency. The purpose of this calculation is to normalize the spatial difference according to a geometrically distributed range of scales, making it suitable for sinusoidal transformation.

#### Parameters
- `coord` (float): The spatial difference (deltaX or deltaY) between two points.
- `cur_freq` (int): The current frequency index being applied, which determines the scale of the encoding.

#### Returns
- `angle` (float): The calculated angle for the given coordinate at the specified frequency. This angle is used to generate sinusoidal function values (sin and cos).

### `cal_coord_embed(self, coords_tuple)`
#### Description
The `cal_coord_embed` function transforms a tuple of spatial coordinates into a high-dimensional vector using sinusoidal encoding. Each coordinate in the tuple is processed with multiple sinusoidal frequencies, and the results from the sine and cosine of each frequency are concatenated to form the final embedding vector.

#### Parameters
- `coords_tuple` (Tuple[float, float]): A tuple containing the spatial differences (deltaX, deltaY) between points.

#### Returns
- `embed` (List[float]): A list representing the high-dimensional spatial embedding. The length of this list is 2 `frequency_num` coord_dim, containing interleaved sine and cosine values for each frequency and coordinate.
