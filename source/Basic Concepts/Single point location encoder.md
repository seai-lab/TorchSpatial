---
output:
  html_document: default
  pdf_document: default
---

# 1. Single point location encoder
<p align="center">
  <img src="../images/single_location_encoder_structure.png" alt="Location Encoder Structure" title="General Structure of Single Location Encoder" width="30%" />
</p>


## 1.1 EncoderMultiLayerFeedForwardNN()  
`NN(⋅) : ℝ^W -> ℝ^d` is a learnable neural network component which maps the input position embedding `PE(x) ∈ ℝ^W` into the location embedding `Enc(x) ∈ ℝ^d`. A common practice is to define `NN(⋅)` as a multi-layer perceptron, while Mac Aodha et al. (2019) adopted a more complex `NN(⋅)` which includes an initial fully connected layer, followed by a series of residual blocks. The purpose of `NN(⋅)` is to provide a learnable component for the location encoder, which captures the complex interaction between input locations and target labels.

### 1.1.1 Properties

- `input_dim` (int): Dimensionality of the input embeddings.
- `output_dim` (int): Dimensionality of the output of the network.
- `num_hidden_layers` (int): The number of hidden layers in the network. If set to 0, the network will be linear.
- `dropout_rate` (float, optional): The dropout rate for regularization. If None, dropout is not used.
- `hidden_dim` (int): The size of each hidden layer. Required if `num_hidden_layers` is greater than 0.
- `activation` (str): The type of activation function to use in the hidden layers. Common options are 'sigmoid', 'tanh', or 'relu'.
- `use_layernormalize` (bool): Determines whether to apply layer normalization after each hidden layer.
- `skip_connection` (bool): If set to True, enables skip connections between layers.
- `context_str` (str, optional): An optional string providing context for this instance, such as indicating its role within a larger model.

### 1.1.3 Methods

#### `__init__(input_dim, output_dim, num_hidden_layers=0, dropout_rate=None, hidden_dim=-1, activation="sigmoid", use_layernormalize=False, skip_connection=False, context_str=None)`
Constructor for the `EncoderMultiLayerFeedForwardNN` class.

- **Parameters**:
  - `input_dim` (int): Dimensionality of the input embeddings.
  - `output_dim` (int): Dimensionality of the output of the network.
  - `num_hidden_layers` (int): Number of hidden layers in the network, set to 0 for a linear network.
  - `dropout_rate` (float, optional): Dropout keep probability.
  - `hidden_dim` (int): Size of the hidden layers.
  - `activation` (str): Activation function to use ('tanh' or 'relu').
  - `use_layernormalize` (bool): Whether to use layer normalization.
  - `skip_connection` (bool): Whether to use skip connections.
  - `context_str` (str, optional): Contextual string for the encoder.

#### `forward(input_tensor)`
Defines the forward pass of the network.

- **Parameters**:
  - `input_tensor` (Tensor): A tensor with shape `[batch_size, ..., input_dim]`.
- **Returns**: A tensor with shape `[batch_size, ..., output_dim]`. Note that no non-linearity is applied to the output.

- **Raises**:
  - `AssertionError`: If the last dimension of `input_tensor` does not match `input_dim`.



## 1.2 PositionEncoder()
`PE(⋅)` is the most important component which distinguishes different `Enc(x)`. Usually, `PE(⋅)` is a *deterministic* function which transforms location x into a W-dimension vector, so-called position embedding. The purpose of `PE(⋅)` is to do location feature normalization (Chu et al. 2019, Mac Aodha et al. 2019, Rao et al. 2020) and/or feature decomposition (Mai et al. 2020b, Zhong et al. 2020) so that the output `PE(x)` is more learning-friendly for `NN(⋅)`. In Table 1 we further classify different `Enc(x)` into four sub-categories based on their `PE(⋅)`: discretization-based, direct, sinusoidal, and sinusoidal multi-scale location encoder. Each of them will be discussed in detail below.

### 1.2.1 Properties
- `spa_embed_dim` (int): The dimension of the output spatial relation embedding.
- `coord_dim` (int): The dimensionality of space (e.g., 2 for 2D, 3 for 3D).
- `frequency_num` (int): The number of different frequencies/wavelengths for the sinusoidal functions.
- `max_radius` (float): The largest context radius the model can handle.
- `min_radius` (float): The smallest context radius considered by the model.
- `freq_init` (str): Method to initialize the frequency list ('random' or 'geometric').
- `ffn` (nn.Module, optional): A feedforward neural network module to be applied to the embeddings.
- `device` (str): The device to which tensors will be moved ('cuda' or 'cpu').


### 1.2.2 Methods
### `get_activation_function(activation, context_str)`
- **Parameters**:
  - `activation`: A string that specifies the type of activation function to retrieve.
  - `context_str`: A string that provides context for the error message if the activation function is not recognized.
- **Returns**: An activation function object from the `torch.nn` module.
- **Description**: Retrieves an activation function object based on the specified `activation` string. It supports 'leakyrelu', 'relu', 'sigmoid', and 'tanh'. If the specified activation is not recognized, it raises an exception with a context-specific error message.
- **Exceptions**: Raises an `Exception` with the message `"{context_str} activation not recognized."` if the specified activation function is not one of the supported options.


#### `cal_freq_list(freq_init, frequency_num, max_radius, min_radius)`
- **Parameters**:
  - `freq_init`: A string that specifies the initialization method for frequencies ('random' or 'geometric').
  - `frequency_num`: An integer representing the number of frequencies to generate.
  - `max_radius`: A float representing the maximum radius, used as the upper bound for random initialization or the geometric sequence's start point.
  - `min_radius`: A float representing the minimum radius, used as the geometric sequence's end point.
- **Returns**: A NumPy array `freq_list` containing the list of frequencies initialized as per the method specified by `freq_init`.
- **Description**: Calculates a list of frequencies based on the initialization method specified. If `freq_init` is 'random', it generates `frequency_num` random frequencies, each multiplied by `max_radius`. If `freq_init` is 'geometric', it generates a list of frequencies based on a geometric progression from `min_radius` to `max_radius` with `frequency_num` elements.
- **Exceptions**: None explicitly raised, but if `frequency_num` is less than 1, it may cause an error in the geometric initialization logic.


#### `cal_freq_mat()`
Generates a matrix of frequencies for encoding.
- **Returns**: A frequency matrix (`np.array`) for use in positional encoding.

#### `cal_input_dim()`
Computes the dimension of the encoded spatial relation embedding based on the frequency and coordinate dimensions.
- **Returns**: The input dimension (int) of the encoder.

#### `cal_elementwise_angle(coord, cur_freq)`
Calculates the angle for each coordinate and frequency, to be used in the sinusoidal functions.
- **Parameters**:
  - `coord`: The coordinate value (`deltaX` or `deltaY`).
  - `cur_freq`: The current frequency being processed.
- **Returns**: The calculated angle (float).

#### `cal_coord_embed(coords_tuple)`
Encodes a tuple of coordinates into a sinusoidal embedding.
- **Parameters**:
  - `coords_tuple`: A tuple of coordinate values.
- **Returns**: A list of sinusoidal embeddings (`list`).

#### `forward(coords)`
Abstract method for transforming spatial coordinates into embeddings. Must be implemented by subclasses.
- **Parameters**:
  - `coords`: Spatial coordinates to encode.
- **Raises**:
  - `NotImplementedError`: If the method is not overridden by a subclass.

#### `visualize_embed_cosine`
Visualizes the cosine similarity of embeddings on a 2D plot.
- **Parameters**:
  - `embed`: Embedding vector with shape `(spa_embed_dim, 1)`.
  - `module`: The model module containing the embedding layers.
  - `layername`: Specifies the layer name for which the embeddings are visualized (`"input_emb"` or `"output_emb"`).
  - `coords`: Coordinates for the embeddings.
  - `extent`: Extent of the plot area.
  - `centerpt`: (Optional) The center point to highlight.
  - `xy_list`: (Optional) List of points to plot.
  - `pt_size`: (Optional) Size of the points.
  - `polygon`: (Optional) Polygon to outline on the plot.
  - `img_path`: (Optional) Path to save the plot image.

#### `get_coords`
Generates a grid of coordinates within a specified extent.
- **Parameters**:
  - `extent`: The bounding box for the coordinate grid.
  - `interval`: The spacing between points in the grid.

#### `map_id2geo`
Plots geographical locations based on their IDs.
- **Parameters**:
  - `place2geo`: A mapping from place IDs to geographical coordinates.

#### `visualize_encoder`
Visualizes the output of an encoder layer for a given set of coordinates.
- **Parameters**:
  - `module`: The model module containing the encoder.
  - `layername`: Specifies the encoder layer (`"input_emb"` or `"output_emb"`).
  - `coords`: Coordinates for visualization.
  - `extent`: Extent of the plot area.
  - `num_ch`: Number of channels to visualize.
  - `img_path`: (Optional) Path to save the visualization.

#### `spa_enc_embed_clustering`
Performs spatial encoding embedding clustering and visualization.
- **Parameters**:
  - `module`: The model module to use for forward pass.
  - `num_cluster`: Number of clusters for the agglomerative clustering.
  - `extent`: Extent of the plot area.
  - `interval`: Interval between points in the grid.
  - `coords`: Coordinates for clustering.
  - `tsne_comp`: Number of components for t-SNE reduction.

#### `make_enc_map`
Creates a map visualization based on encoder cluster labels.
- **Parameters**:
  - `cluster_labels`: Cluster labels for each point in the grid.
  - `num_cluster`: Number of clusters.
  - `extent`: Extent of the plot area.
  - `margin`: Margin around the plot area.
  - `xy_list`: (Optional) List of points to plot.
  - `polygon`: (Optional) Polygon to outline on the plot.
  - `usa_gdf`: (Optional) GeoDataFrame for the USA map.
  - `coords_color`: (Optional) Color for the coordinates.
  - `colorbar`: (Optional) Flag to display a color bar.
  - `img_path`: (Optional) Path to save the map image.
  - `xlabel`, `ylabel`: (Optional) Labels for the x and y axes.

#### `explode`
Converts a GeoDataFrame with MultiPolygons into a GeoDataFrame with Polygons.
- **Parameters**:
  - `indata`: Input GeoDataFrame or file path.

#### `get_pts_in_box`
Filters points within a specified bounding box.
- **Parameters**:
  - `place2geo`: A mapping from place IDs to geographical coordinates.
  - `extent`: The bounding box for filtering.

#### `load_USA_geojson`
Loads and projects the USA mainland GeoJSON to the EPSG:2163 projection system.
- **Parameters**:
  - `us_geojson_file`: Path to the USA GeoJSON file.

#### `get_projected_mainland_USA_states`
Loads and projects mainland USA states from a GeoJSON file to the EPSG:2163 projection system.
- **Parameters**:
  - `us_states_geojson_file`: Path to the USA states GeoJSON file.

#### `read2idIndexFile`
Reads an entity or relation to ID mapping file.
- **Parameters**:
  - `Index2idFilePath`: Path to the file containing the mappings.

#### `reverse_dict`
Reverses a dictionary mapping.
- **Parameters**:
  - `iri2id`: The dictionary to reverse.

#### `get_node_mode`
Determines the mode (type) of a node based on the provided mappings.
- **Parameters**:
  - `node_maps`: A mapping of node types to their IDs.
  - `node_id`: The ID of the node to determine the mode for.

#### `path_embedding_compute`
Computes the embedding for a path between nodes.
- **Parameters**:
  - `path_dec`: The path decoder.


<div style="display:none">
# 2. Aggregation location encoder
<p align="center">
  <img src="./figs/aggregation_location_encoder_structure.png" alt="Structure of Aggregation Location Encoder Structure" title="General Structure of Location Encoder" width="40%" />
</p>
</div>
