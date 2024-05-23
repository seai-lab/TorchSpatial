import pickle
import torch
from collections import OrderedDict, defaultdict
import random
import json

import numpy as np




def coord_normalize(coords, extent = (-180, 180, -90, 90)):
    """
    Given a list of coords (X, Y), normalize them to [-1, 1]
    Args:
        coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        extent: (x_min, x_max, y_min, y_max)
    Return:
        coords_mat: np tensor shape (batch_size, num_context_pt, coord_dim)
    """
    if type(coords) == list:
        coords_mat = np.asarray(coords).astype(float)
    elif type(coords) == np.ndarray:
        coords_mat = coords

    # x => [0,1]  min_max normalize
    x = (coords_mat[:,:,0] - extent[0])*1.0/(extent[1] - extent[0])
    # x => [-1,1]
    coords_mat[:,:,0] = (x * 2) - 1

    # y => [0,1]  min_max normalize
    y = (coords_mat[:,:,1] - extent[2])*1.0/(extent[3] - extent[2])
    # x => [-1,1]
    coords_mat[:,:,1] = (y * 2) - 1

    return coords_mat

def json_load(filepath):
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return data

def json_dump(data, filepath, pretty_format = True):
    with open(filepath, 'w') as fw:
        if pretty_format:
            json.dump(data, fw, indent=2, sort_keys=True)
        else:
            json.dump(data, fw)

def pickle_dump(obj, pickle_filepath):
    with open(pickle_filepath, "wb") as f:
        pickle.dump(obj, f, protocol=2)

def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj