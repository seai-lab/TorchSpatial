import numpy as np
from sklearn.neighbors import KDTree, BallTree

from scipy import sparse
from scipy.spatial.distance import cdist

from spatial_self_information import AnalyticalSurprisal
import torch
from tqdm import tqdm


def add_random_background(data, locations, n_bg=400):
    n = data.shape[0]
    n_compliment = n_bg - n

    assert n_compliment / n_bg > 0.7, "Foreground too dense!"

    lon_max, lat_max = np.max(locations, axis=0)
    lon_min, lat_min = np.min(locations, axis=0)

    random_lons = lon_min + np.random.rand(n_compliment) * (lon_max - lon_min)
    random_lats = lat_min + np.random.rand(n_compliment) * (lat_max - lat_min)
    random_locations = np.concatenate((random_lons.reshape(-1, 1), random_lats.reshape(-1, 1)), axis=1)

    data = np.concatenate((data, np.zeros(n_compliment)))
    locations = np.concatenate((locations, random_locations))

    return data, locations

def compute_spatial_self_information(data, locations, k=4):
    analytical_surprisal = AnalyticalSurprisal()

    kdt = BallTree(locations, leaf_size=30, metric='haversine')
    dists, nbrs = kdt.query(locations, k=k, return_distance=True)

    weights, coord_is, coord_js = [], [], []

    for i, nis in enumerate(nbrs):
        weights += [1 for _ in range(len(nis[1:]))]
        coord_is += [i for _ in range(len(nis[1:]))]
        coord_js += [j for j in nis[1:]]

        weights += [1 for _ in range(len(nis[1:]))]
        coord_is += [j for j in nis[1:]]
        coord_js += [i for _ in range(len(nis[1:]))]

    d = locations.shape[0]
    weight = sparse.coo_matrix((weights, (coord_is, coord_js)), shape=(d, d)).toarray()
    weight[weight > 0] = 1

    base_data = np.copy(data)
    base_data[base_data != 0] = np.mean(data[data != 0])

    cs, ns = np.unique(base_data, return_counts=True)
    rmax = np.argmax(ns)

    ignores = np.ones_like(cs)
    ignores[rmax] = 0

    analytical_surprisal.fit(cs, ns, weight, ignores)

    prob = analytical_surprisal.get_probability(data, weight)

    base_info = -np.log(prob[0])

    shuffle_data = np.copy(data)
    shuffle_infos = []
    for i in range(30):
        shuffle_data[shuffle_data != 0] = np.random.permutation(shuffle_data[shuffle_data != 0])

        cs, ns = np.unique(shuffle_data, return_counts=True)
        rmax = np.argmax(ns)

        ignores = np.ones_like(cs)
        ignores[rmax] = 0

        analytical_surprisal.fit(cs, ns, weight, ignores)

        prob = analytical_surprisal.get_probability(shuffle_data, weight)

        shuffle_infos.append(-np.log(prob[0]))

    shuffle_info = np.min(shuffle_infos)

    cs, ns = np.unique(data, return_counts=True)
    bg_rate = np.max(ns) / np.sum(ns)
    rmax = np.argmax(ns)

    ignores = np.ones_like(cs)
    ignores[rmax] = 0

    analytical_surprisal.fit(cs, ns, weight, ignores)

    prob = analytical_surprisal.get_probability(data, weight)

    performance_info = -np.log(prob[0])

    return bg_rate, base_info, shuffle_info, performance_info

def compute_spatial_self_information_for_dataset(cosines, locations, dists, nbrs, radius, n_bg):
    bg_rates, base_infos, shuffle_infos, sample_infos = [], [], [], []

    for nbr_idxs, nbr_dists in tqdm(zip(nbrs, dists), total=locations.shape[0]):
        dist_idxs = np.where(nbr_dists < radius)
        nbr_idxs = nbr_idxs[dist_idxs]

        if len(nbr_idxs) < 5:
            bg_rate, base_info, shuffle_info, sample_info = 1, 0, 0, 0
        else:
            plot_data, plot_locations = add_random_background(cosines[nbr_idxs], locations[nbr_idxs], n_bg)
            bg_rate, base_info, shuffle_info, sample_info = compute_spatial_self_information(plot_data, plot_locations)

        bg_rates.append(bg_rate)
        base_infos.append(base_info)
        shuffle_infos.append(shuffle_info)
        sample_infos.append(sample_info)

    base_infos = np.array(base_infos)
    shuffle_infos = np.array(shuffle_infos)
    sample_infos = np.array(sample_infos)

    base_infos[base_infos > 1e8] = 1e8
    shuffle_infos[shuffle_infos > 1e8] = 1e8
    sample_infos[sample_infos > 1e8] = 1e8

    return base_infos, shuffle_infos, sample_infos

def info_to_prob(infos, inverse):
    if inverse:
        tmp = np.exp(infos/100)
    else:
        tmp = np.exp(-infos/100)
    return tmp / np.sum(tmp)

def ssi_downsample(infos, sample_rate=0.9, inverse=False):
    return np.random.choice(len(infos), size=int(len(infos) * sample_rate), p=info_to_prob(infos, inverse), replace=False)

def ssi_sample(features, locations, sample_rate, k=20, radius=100, n_bg=100, bucket_size=0.1, inverse=False):
    locations = (locations / np.array([180, 90])) * np.array([np.pi, np.pi / 2])

    dist_cosine = cdist(features, np.mean(features, axis=0, keepdims=True), metric="cosine").flatten() // bucket_size

    kdt = BallTree(locations, leaf_size=30, metric='haversine')
    dists, nbrs = kdt.query(locations, k=k, return_distance=True)

    dists *= 6371

    base_infos, shuffle_infos, sample_infos = compute_spatial_self_information_for_dataset(dist_cosine, locations, dists, nbrs, radius, n_bg)
    idx = ssi_downsample(base_infos, sample_rate=sample_rate, inverse=inverse)

    return idx
