import numpy as np
from datetime import datetime


def merge_proba(probas, n_outputs):

    n_samples, n_features = probas.shape

    if n_features % n_outputs != 0:
        msg = "The dimension of probas = {} does not match n_outputs = {}."
        raise RuntimeError(msg.format(n_features, n_outputs))

    proba = np.mean(probas, axis=1)
    proba = proba.reshape([n_samples, 1])
    return proba


def init_array(X, n_aug_features):

    n_samples, n_features = X.shape
    n_dims = n_features + n_aug_features
    X_middle = np.zeros((n_samples, n_dims))   #, dtype=np.uint8
    X_middle[:, : n_features] += X

    return X_middle


def merge_array(X_middle, X_aug, n_features):

    assert X_middle.shape[0] == X_aug.shape[0]  # check n_samples
    assert X_middle.shape[1] == n_features + X_aug.shape[1]  # check n_features
    X_middle[:, n_features:] = X_aug

    return X_middle


def ctime():

    ctime = '[' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ']'
    return ctime
