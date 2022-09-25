import scipy.io
import numpy as np
import pandas as pd
import torch
import utility.utils as utils
import os
import json
import networkx as nx
from networkx.readwrite import json_graph


def load_dataset(name, sparse=False, tensor=True, size_limit=0, device='cpu'):
    try:
        path = 'data/' + name + '_dataset.mat'
        task = scipy.io.loadmat(path)['task']
    except FileNotFoundError:
        path = 'data/' + name.upper() + '_dataset.mat'
        task = scipy.io.loadmat(path)['task']

    info = task['info'].flat[0]
    input_dim, k = int(info['U']), float(info['k'])

    folds = task['folds'][0][0]     # [n_folds, 1]

    tr_samples, test_samples, nested_trs, nested_vals = [], [], [], []
    for fold in folds:
        fold = fold[0][0]
        training = fold['training'][0][0] - 1      # [1, n_training_graphs]
        test = fold['test'][0][0] - 1              # [1, n_test_graphs]
        nestedtr = fold['nestedtr'][0]
        validation = fold['validation'][0]
        nested_tr = []
        nested_val = []
        for i in range(nestedtr.shape[1]):
            tr_inside = nestedtr[0][i][0] - 1
            val_inside = validation[0][i][0] - 1
            nested_tr.append(tr_inside)
            nested_val.append(val_inside)

        tr_samples.append(training)
        test_samples.append(test)
        nested_trs.append(nested_tr)
        nested_vals.append(nested_val)

    data = task['data']
    inputs = data[0][0]['input'].flat[0]     # [n_graph, 1]
    target = data[0][0]['target'].flat[0][0]   # [1, n_graph]
    if name == 'collab':
        target = data[0][0]['target'][0][0]

    adj_mats = []
    node_labels = []
    for graph in inputs:
        adj_mat = graph[0]['adjacency_matrix'].flat[0]
        label = graph[0]['labels'].flat[0]

        if not sparse:
            if type(adj_mat) is not np.ndarray:
                adj_mat = adj_mat.toarray()

        if tensor:
            adj_mat = torch.tensor(adj_mat, dtype=torch.double, device=device)
            label = torch.tensor(label, dtype=torch.double, device=device)

        adj_mats.append(adj_mat)
        node_labels.append(label)

    # Reduce the number of graphs
    if size_limit:

        assert size_limit <= len(adj_mats), 'Size limit is not effective.'

        rand_seed = 1
        np.random.seed(rand_seed)
        idx = np.random.randint(0, len(adj_mats), size_limit)

        adj_mats = [adj_mats[i] for i in idx]
        node_labels = [node_labels[i] for i in idx]
        target = target[:, idx].T  # TODO, this applies to onehot only

        # TODO: may have a better way to trim dataset instead of re-making cross-validation splits

        tr_samples, test_samples = utils.cross_validation_split(
            list(range(size_limit)), num_fold=10, shuffle=False)
    else:
        idx = np.arange(len(adj_mats))

    return input_dim, k, adj_mats, node_labels, target, tr_samples, test_samples, nested_trs, nested_vals, idx
