# FCNet

## Overview
The `FCNet` class is a fully connected neural network designed for classification tasks. It consists of multiple residual layers to enhance feature extraction and can classify input data into multiple categories.

## Features
- **Fully Connected Layers**: Utilizes linear layers to transform input data.
- **Residual Layers**: Includes residual layers to improve feature extraction.
- **Class Embedding**: Embeds the extracted features into the desired number of classes.
- **User Embedding**: Optionally embeds features into user-specific embeddings.

## Configuration Parameters
- `num_inputs`: Dimensionality of the input embedding.
- `num_classes`: Number of categories for classification.
- `num_filts`: Dimensionality of the hidden embeddings.
- `num_users`: Number of user-specific embeddings (default is 1).

## Methods
### `forward(x, class_of_interest=None, return_feats=False)`
- **Purpose**: Processes input features through the network and returns class predictions or intermediate feature embeddings.
- **Parameters**:
  - `x` (torch.FloatTensor): Input location features with shape `(batch_size, input_loc_dim)`.
  - `class_of_interest` (int, optional): Class ID for specific class evaluation.
  - `return_feats` (bool, optional): Whether to return only the intermediate feature embeddings.
- **Returns**:
  - If `return_feats` is True, returns the feature embeddings with shape `(batch_size, num_filts)`.
  - If `class_of_interest` is specified, returns the sigmoid output for the specific class with shape `(batch_size)`.
  - Otherwise, returns the sigmoid class predictions for all classes with shape `(batch_size, num_classes)`.

### `eval_single_class(x, class_of_interest)`
- **Purpose**: Evaluates the network output for a specific class.
- **Parameters**:
  - `x` (torch.FloatTensor): Feature embeddings with shape `(batch_size, num_filts)`.
  - `class_of_interest` (int): Class ID for evaluation.
- **Returns**:
  - Sigmoid output for the specified class with shape `(batch_size)`.



### User Embedding
The user embedding layer maps the hidden features to user-specific embeddings:

$\mathbf{y} _{\text{user}} = \mathbf{W} _{\text{user}} \mathbf{h}$

where $\mathbf{W}_{\text{user}}$ is the user weight matrix.
