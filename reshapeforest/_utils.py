import numpy as np
from datetime import datetime


def merge_proba(probas, n_outputs):
    """
    ***only useful for one class, anomaly detection
    average the results of "n" different forests, finally, it's a two-dimensional matrix
    (the features represents the probability that one data belongs to different classes).

    Merge an array that stores multiple class distributions from all estimators
    in a cascade layer into a final class distribution."""

    n_samples, n_features = probas.shape

    if n_features % n_outputs != 0:
        msg = "The dimension of probas = {} does not match n_outputs = {}."
        raise RuntimeError(msg.format(n_features, n_outputs))

    proba = np.mean(probas, axis=1)
    proba = proba.reshape([n_samples, 1])
    return proba


def init_array(X, n_aug_features):
    """
    add "0" to the end of the original features...
    Initialize a array that stores the intermediate data used for training
    or evaluating the model."""
    # if X.dtype != np.uint8:
    #     msg = "The input `X` when creating the array should be binned."
    #     raise ValueError(msg)

    # Create the global array that stores both X and X_aug
    n_samples, n_features = X.shape
    n_dims = n_features + n_aug_features
    X_middle = np.zeros((n_samples, n_dims))   #, dtype=np.uint8
    X_middle[:, : n_features] += X

    return X_middle


def merge_array(X_middle, X_aug, n_features):
    """
    add the output label to the end of the original features...
    Update the array created by `init_array`  with additional checks on the
    layout."""

    # if X_aug.dtype != np.uint8:
    #     msg = "The input `X_aug` when merging the array should be binned."
    #     raise ValueError(msg)

    assert X_middle.shape[0] == X_aug.shape[0]  # check n_samples
    assert X_middle.shape[1] == n_features + X_aug.shape[1]  # check n_features
    X_middle[:, n_features:] = X_aug

    return X_middle


def ctime():
    """A formatter on current time used for printing running status."""
    ctime = '[' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ']'
    return ctime
