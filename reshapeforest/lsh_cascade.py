"""Implementation of Deep LSH based Isolation Forest."""

__all__ = ["CascadeLSHForest"]

import time
import numbers
import numpy as np
from abc import ABCMeta, abstractmethod

from . import _utils
from . import _io
from .lsh_layer import Layer

class BaseCascadeLSHForest(metaclass=ABCMeta):

    def __init__(
            self,
            label_features = False,
            embedding_features = False,
            both_features = True,
            max_layers=3,  #set****************(cascade layers 3, not including the last predict layer)
            n_estimators=3, # modify it according to the real estimators
            n_trees=100,
            granularity=1,

            predictor="ALSHforest",
            n_tolerant_rounds=2,
            #delta=1e-5,
            partial_mode=False,
            n_jobs=None,
            random_state=None,
            verbose=1
    ):

        self.max_layers = max_layers
        self.n_estimators = n_estimators
        self.n_trees = n_trees

        self.granularity = granularity,
        self.n_tolerant_rounds = n_tolerant_rounds
        self.partial_mode = partial_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Utility variables
        self.n_layers_ = 0
        self.is_fitted_ = False

        # Internal containers
        self.layers_ = {}
        self.buffer_ = _io.Buffer(partial_mode)

        # select concatenated features
        self.label_features = label_features
        self.embedding_features = embedding_features
        self.both_features = both_features

    def __len__(self):
        return self.n_layers_

    def __getitem__(self, index):
        return self._get_layer(index)


    def _get_layer(self, layer_idx):
        if not 0 <= layer_idx <= self.n_layers_:
            msg = (
                "The layer index should be in the range [0, {}], but got {}"
                " instead."
            )
            raise ValueError(msg.format(self.n_layers_ - 1, layer_idx))

        layer_key = "layer_{}".format(layer_idx)

        return self.layers_[layer_key]

    def _set_layer(self, layer_idx, layer):
        layer_key = "layer_{}".format(layer_idx)
        if layer_key in self.layers_:
            msg = ("Layer with the key {} already exists in the internal"
                   " container.")
            raise RuntimeError(msg.format(layer_key))

        self.layers_.update({layer_key: layer})

    def _set_n_trees(self, layer_idx):
        """
        Set the number of isolation trees for each estimator.
        """
        # The number of trees for each layer is fixed as `n_trees`.
        if isinstance(self.n_trees, numbers.Integral):
            if not self.n_trees > 0:
                msg = "n_trees = {} should be strictly positive."
                raise ValueError(msg.format(self.n_trees))
            return self.n_trees

        elif self.n_trees == "auto":
            n_trees = 100 * (layer_idx + 1)
            return n_trees if n_trees <= 500 else 500
        else:
            msg = ("Invalid value for n_trees. Allowed values are integers or"
                   " 'auto'.")
            raise ValueError(msg)

    def _check_input(self, X):
        """
        Check the input data and set the attributes if X is training data."""
        is_training_data = X is not None

        if is_training_data:
            _, self.n_features_ = X.shape
            self.n_outputs_ = 1   # the number of label types

    @property
    def n_aug_features_(self):
        return 2 * self.n_estimators * self.n_outputs_

    def fit(self, X):
        """
        Build the DeepiForest using the training data.
        """
        self._check_input(X)

        X_train_ = X
        X_train_ = self.buffer_.cache_data(0, X_train_, is_training_data=True)

        # =====================================================================
        # Training Stage
        # =====================================================================

        if self.verbose > 0:
            print("{} Start to fit the model:".format(_utils.ctime()))

        # Build the first cascade layer
        layer_ = Layer(
            0,
            self.n_outputs_,
            self.n_estimators,
            self._set_n_trees(0),
            self.granularity,
            self.partial_mode,
            self.buffer_,
            self.n_jobs,
            self.verbose
        )

        if self.verbose > 0:
            print("{} Fitting cascade layer = {:<2}".format(_utils.ctime(), 0))

        tic = time.time()
        layer_.fit_transform(X_train_)
        if self.label_features:
            label_middle_features = layer_.transform(X_train_)
        elif self.embedding_features:
            embedding_middle_features = layer_.path_train(X_train_)
        else:
            label_features = layer_.transform(X_train_)
            embedding_features = layer_.path_train(X_train_)
            both_middle_features = np.concatenate((label_features, embedding_features),axis=1)

        toc = time.time()
        training_time = toc - tic

        if self.verbose > 0:
            msg = "{} layer = {:<2} | Elapsed = {:.3f} s"
            print(
                msg.format(
                    _utils.ctime(),
                    0,
                    training_time
                )
            )

        # Add the first cascade layer
        self._set_layer(0, layer_)
        self.n_layers_ += 1

        # ====================================================================
        # Main loop on the training stage
        # ====================================================================

        while self.n_layers_ < self.max_layers:

            #X_middle_train_ = _utils.merge_array(X_middle_train_, X_binned_aug_train_, self.n_features_)
            if self.label_features:
                X_middle_train_ = np.concatenate((X_train_, label_middle_features),axis=1)
            elif self.embedding_features:
                X_middle_train_ = np.concatenate((X_train_, embedding_middle_features),axis=1)
            else:
                X_middle_train_ = np.concatenate((X_train_, both_middle_features),axis=1)

            # Build a cascade layer
            layer_idx = self.n_layers_
            layer_ = Layer(
                layer_idx,
                self.n_outputs_,
                self.n_estimators,
                self._set_n_trees(layer_idx),
                self.granularity,
                self.partial_mode,
                self.buffer_,
                self.n_jobs,
                self.verbose
            )

            X_middle_train_ = self.buffer_.cache_data(
                layer_idx, X_middle_train_, is_training_data=True
            )

            if self.verbose > 0:
                msg = "{} Fitting cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            tic = time.time()
            layer_.fit_transform(X_middle_train_)
            if self.label_features:
                label_middle_features = layer_.transform(X_middle_train_)
            elif self.embedding_features:
                embedding_middle_features = layer_.path_train(X_middle_train_)
            else:
                label_features = layer_.transform(X_middle_train_)
                embedding_features = layer_.path_train(X_middle_train_)
                both_middle_features = np.concatenate((label_features, embedding_features), axis=1)

            toc = time.time()
            training_time = toc - tic

            if self.verbose > 0:
                msg = "{} layer = {:<2} | Elapsed = {:.3f} s"
                print(
                    msg.format(
                        _utils.ctime(),
                        layer_idx,
                        training_time
                    )
                )

            # Update the cascade layer
            self._set_layer(layer_idx, layer_)
            self.n_layers_ += 1

        # ====================================================================
        # Last prediction layer on the training stage
        # ====================================================================
        if self.n_layers_ == self.max_layers and self.verbose > 0:
            msg = "{} Reaching the maximum number of layers: {}"
            print(msg.format(_utils.ctime(), self.max_layers))

            if self.label_features:
                X_middle_train_ = np.concatenate((X_train_, label_middle_features),axis=1)
            elif self.embedding_features:
                X_middle_train_ = np.concatenate((X_train_, embedding_middle_features),axis=1)
            else:
                X_middle_train_ = np.concatenate((X_train_, both_middle_features),axis=1)

            # Build a cascade layer
            layer_idx = self.n_layers_
            layer_ = Layer(
                layer_idx,
                self.n_outputs_,
                self.n_estimators,
                self._set_n_trees(layer_idx),
                self.granularity,
                self.partial_mode,
                self.buffer_,
                self.n_jobs,
                self.verbose
            )

            X_middle_train_ = self.buffer_.cache_data(
                layer_idx, X_middle_train_, is_training_data=True
            )

            if self.verbose > 0:
                msg = "{} Fitting cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            tic = time.time()
            layer_.fit_transform(X_middle_train_)
            toc = time.time()
            training_time = toc - tic

            if self.verbose > 0:
                msg = "{} layer = {:<2} | Elapsed = {:.3f} s"
                print(
                    msg.format(
                        _utils.ctime(),
                        layer_idx,
                        training_time
                    )
                )

            # Update the cascade layer
            self._set_layer(layer_idx, layer_)
            self.n_layers_ += 1

        self.is_fitted_ = True

        return self


## Implementation of the CascadeLSHForest.

class CascadeLSHForest(BaseCascadeLSHForest):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _repr_performance(self, pivot):
        msg = "Val Acc = {:.3f} %"
        return msg.format(pivot * 100)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        """
        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        X_test = X

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)
            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                if self.label_features:
                    label_middle_features = layer.transform(X_test)
                elif self.embedding_features:
                    embedding_middle_features = layer.path_test(X_test)
                else:
                    label_features = layer.transform(X_test)
                    embedding_features = layer.path_test(X_test)
                    both_middle_features = np.concatenate((label_features, embedding_features), axis=1)
            elif layer_idx < self.n_layers_-1:
                if self.label_features:
                    X_middle_test_ = np.concatenate((X_test, label_middle_features), axis=1)
                    label_middle_features = layer.transform(X_middle_test_)
                elif self.embedding_features:
                    X_middle_test_ = np.concatenate((X_test, embedding_middle_features), axis=1)
                    embedding_middle_features = layer.path_test(X_middle_test_)
                else:
                    X_middle_test_ = np.concatenate((X_test, both_middle_features), axis=1)
                    label_features = layer.transform(X_middle_test_)
                    embedding_features = layer.path_test(X_middle_test_)
                    both_middle_features = np.concatenate((label_features, embedding_features), axis=1)
            else:
                if self.label_features:
                    X_middle_test_ = np.concatenate((X_test, label_middle_features), axis=1)
                elif self.embedding_features:
                    X_middle_test_ = np.concatenate((X_test, embedding_middle_features), axis=1)
                else:
                    X_middle_test_ = np.concatenate((X_test, both_middle_features), axis=1)

        if self.verbose > 0:
            msg = "{} Final evaluating layer = {:<2}"
            print(msg.format(_utils.ctime(), self.n_layers_-1))
        layer = self._get_layer(self.n_layers_-1)
        proba = layer.predict_full(X_middle_test_)
        proba = _utils.merge_proba(proba, self.n_outputs_)

        return proba

    def predict(self, X):
        """
        Predict class for X.
        """
        proba = self.predict_proba(X)
        a = np.array(proba)

        return a[:,0]
