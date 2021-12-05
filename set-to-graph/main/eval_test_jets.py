import torch
import uproot3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics

from dataloaders.jets_loader import JetGraphDataset
from models import SetPartitionTri


def _get_rand_index(labels, predictions):
    n_items = len(labels)
    if n_items < 2:
        return 1
    n_pairs = (n_items * (n_items - 1)) / 2

    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true) or (not label_true and not pred_true):
                correct_pairs += 1

    return correct_pairs / n_pairs


def _error_count(labels, predictions):
    n_items = len(labels)

    true_positives = 0
    false_positive = 0
    false_negative = 0

    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if label_true and pred_true:
                true_positives += 1
            if (not label_true) and pred_true:
                false_positive += 1
            if label_true and (not pred_true):
                false_negative += 1
    return true_positives, false_positive, false_negative


def _get_recall(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_negative == 0:
        return 0

    return true_positives / (true_positives + false_negative)


def _get_precision(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_positive == 0:
        return 0
    return true_positives / (true_positives + false_positive)


def _f_measure(labels, predictions):
    precision = _get_precision(labels, predictions)
    recall = _get_recall(labels, predictions)

    if precision == 0 or recall == 0:
        return 0

    return 2 * (precision * recall) / (recall + precision)


def eval_jets_on_test_set(model, bsize):

    pred = _predict_on_test_set(model, bsize)

    # you need uproot3 to do this. pip install awkward0 uproot3
    test_ds = uproot3.open('main/data/test/test_data.root')
    jet_df = test_ds['tree'].pandas.df(['jet_flav', 'trk_vtx_index'], flatten=False)
    jet_flav = jet_df['jet_flav']

    target = [x for x in jet_df['trk_vtx_index'].values]

    print('Calculating scores on test set... ', end='')
    start = datetime.now()
    model_scores = {'RI': np.vectorize(_get_rand_index)(target, pred),
                    'ARI': np.vectorize(metrics.adjusted_rand_score)(target, pred),
                    'P': np.vectorize(_get_precision)(target, pred),
                    'R': np.vectorize(_get_recall)(target, pred),
                    'F1': np.vectorize(_f_measure)(target, pred)}

    end = datetime.now()
    print(f': {str(end - start).split(".")[0]}')

    flavours = {5: 'b jets', 4: 'c jets', 0: 'light jets'}
    metrics_to_table = ['P', 'R', 'F1', 'RI', 'ARI']

    df = pd.DataFrame(index=flavours.values(), columns=metrics_to_table)

    for flav_n, flav in flavours.items():
        for metric in metrics_to_table:
            mean_metric = np.mean(model_scores[metric][jet_flav == flav_n])
            df.at[flav, metric] = mean_metric

    return df


def _predict_on_test_set(model, bsize):
    test_ds = JetGraphDataset('test', random_permutation=False)
    model.eval()

    n_tracks = [test_ds[i][0].shape[0] for i in range(len(test_ds))]

    indx_list = []
    predictions = []

    for tracks_in_jet in range(2, np.amax(n_tracks)+1):
        trk_indxs = np.where(np.array(n_tracks) == tracks_in_jet)[0]
        if len(trk_indxs) < 1:
            continue
        indx_list += list(trk_indxs)

        input_list = []
        for iter_idx, i in enumerate(trk_indxs):
            input_list.append(test_ds[i][0])  # shape (1, N_i, 10)
            if (len(input_list) == bsize) or (iter_idx == len(trk_indxs) - 1):
                model_input = torch.stack(input_list)
                if isinstance(model, SetPartitionTri):
                    predictions += list(model(model_input, None)[0].cpu().data.numpy())
                else:
                    edge_vals = model(model_input).squeeze(1)
                    predictions += list(infer_clusters(edge_vals).cpu().data.numpy())  # Shape
                input_list = []

    sorted_predictions = [list(x) for _, x in sorted(zip(indx_list, predictions))]
    return sorted_predictions


def infer_clusters(edge_vals):
    """
    Infer the clusters. Enforce symmetry.
    :param edge_vals: predicted edge score values. shape (B, N, N)
    :return: long tensor shape (B, N) of the clusters.
    """
    # deployment - infer chosen clusters:
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = edge_vals + edge_vals.transpose(1, 2)  # to make symmetric
        pred_matrices = pred_matrices.ge(0.).float()  # adj matrix - 0 as threshold
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.  # each node is always connected to itself
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:  # get connected components - each node connected to all in its component
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
            ones_now = pred_matrices.sum()

        clusters = -1 * torch.ones((b, n), device=edge_vals.device)
        tensor_1 = torch.tensor(1., device=edge_vals.device)
        for i in range(n):
            clusters = torch.where(pred_matrices[:, i] == 1, i * tensor_1, clusters)

    return clusters.long()
