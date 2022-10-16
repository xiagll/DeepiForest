"""Implementation of Deep Forest."""

__all__ = ["CascadeLSHForest"]

import time
import numbers
import numpy as np
from abc import ABCMeta, abstractmethod
from .lsh_forest import ALSHForest, L1SHForest, L2SHForest

from . import _utils
from . import _io


class BaseCascadeLSHForest(metaclass=ABCMeta):

    def __init__(
            self,
            label_features = False,
            embedding_features = True,
            both_features = False,
            n_bins=255,
            bin_subsample=2e5,
            bin_type="percentile",
            max_layers=3,  #set****************(cascade layers, not including the last predict layer)
            n_estimators=2, # modify it according to the real estimators
            n_trees=100,
            granularity=1,

            n_tolerant_rounds=2,
            #delta=1e-5,
            partial_mode=False,
            n_jobs=None,
            random_state=None,
            verbose=1
    ):
        self.n_bins = n_bins
        self.bin_subsample = bin_subsample
        self.bin_type = bin_type
        self.max_layers = max_layers
        self.n_estimators = n_estimators
        self.n_trees = n_trees

        self.granularity = granularity,
        self.n_tolerant_rounds = n_tolerant_rounds
        #self.delta = delta
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
        """Get the layer from the internal container according to the index."""
        if not 0 <= layer_idx <= self.n_layers_:
            msg = (
                "The layer index should be in the range [0, {}], but got {}"
                " instead."
            )
            raise ValueError(msg.format(self.n_layers_ - 1, layer_idx))

        layer_key = "layer_{}".format(layer_idx)

        return self.layers_[layer_key]

    def _set_layer(self, layer_idx, layer):
        """
        Register a layer into the internal container with the given index."""
        layer_key = "layer_{}".format(layer_idx)
        if layer_key in self.layers_:
            msg = ("Layer with the key {} already exists in the internal"
                   " container.")
            raise RuntimeError(msg.format(layer_key))

        self.layers_.update({layer_key: layer})

    def _set_n_trees(self, layer_idx):
        """
        Set the number of decision trees for each estimator in the cascade
        layer with `layer_idx` using the pre-defined rules.
        """
        # The number of trees for each layer is fixed as `n_trees`.
        if isinstance(self.n_trees, numbers.Integral):
            if not self.n_trees > 0:
                msg = "n_trees = {} should be strictly positive."
                raise ValueError(msg.format(self.n_trees))
            return self.n_trees
        # The number of trees for the first 5 layers grows linearly with
        # `layer_idx`, while that for remaining layers is fixed to `500`.
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

    def _validate_params(self):
        """
        Validate parameters, those passed to the sub-modules will not be
        checked here."""

        if not self.max_layers > 0:
            msg = "max_layers = {} should be strictly positive."
            raise ValueError(msg.format(self.max_layers))

        if not self.n_tolerant_rounds > 0:
            msg = "n_tolerant_rounds = {} should be strictly positive."
            raise ValueError(msg.format(self.n_tolerant_rounds))


    @abstractmethod
    def _repr_performance(self, pivot):
        """Format the printting information on training performance."""


    @property
    def n_aug_features_(self):
        return 2 * self.n_estimators * self.n_outputs_

    def fit(self, X):
        """
        Build a deep forest using the training data.
        """
        self._check_input(X)
        self._validate_params()

        X_train_ = X
        X_train_ = self.buffer_.cache_data(0, X_train_, is_training_data=True)

        # =====================================================================
        # Training Stage
        # =====================================================================

        if self.verbose > 0:
            print("{} Start to fit the model:".format(_utils.ctime()))

        # Build the first cascade layer
        layer_ = []

        if self.verbose > 0:
            print("{} Fitting cascade layer = {:<2}".format(_utils.ctime(), 0))

        tic = time.time()
        ALSH = ALSHForest(100, 4)
        ALSH.fit(X_train_)
        layer_.append(ALSH)
        L2SH = L2SHForest(100, 4)
        L2SH.fit(X_train_)
        layer_.append(L2SH)
        # if self.label_features:
        #     label_middle_features = layer_.transform(X_train_)
        if self.embedding_features:
            AL_embedding_middle_features = layer_[0].train_path(X_train_)
            L2_embedding_middle_features = layer_[1].train_path(X_train_)
        #     embedding_middle_features = layer_.path_train(X_train_)
        # else:
        #     label_features = layer_.transform(X_train_)
        #     embedding_features = layer_.path_train(X_train_)
        #     both_middle_features = np.concatenate((label_features, embedding_features),axis=1)

        toc = time.time()
        training_time = toc - tic

        if self.verbose > 0:
            msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
            print(
                msg.format(
                    _utils.ctime(),
                    0,
                    self._repr_performance(1),
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

            AL_middle_train_ = np.concatenate((X_train_, AL_embedding_middle_features),axis=1)
            L2_middle_train_ = np.concatenate((X_train_, L2_embedding_middle_features), axis=1)

            # Build a cascade layer
            layer_idx = self.n_layers_
            layer_ = []

            AL_middle_train_ = self.buffer_.cache_data(
                layer_idx, AL_middle_train_, is_training_data=True
            )
            L2_middle_train_ = self.buffer_.cache_data(
                layer_idx, L2_middle_train_, is_training_data=True
            )
            if self.verbose > 0:
                msg = "{} Fitting cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            tic = time.time()
            ALSH = ALSHForest(100, 4)
            ALSH.fit(L2_middle_train_)
            layer_.append(ALSH)
            L2SH = L2SHForest(100, 4)
            L2SH.fit(AL_middle_train_)
            layer_.append(L2SH)
            if self.embedding_features:
                AL_embedding_middle_features = layer_[0].train_path(L2_middle_train_)
                L2_embedding_middle_features = layer_[1].train_path(AL_middle_train_)

            toc = time.time()
            training_time = toc - tic

            if self.verbose > 0:
                msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
                print(
                    msg.format(
                        _utils.ctime(),
                        layer_idx,
                        self._repr_performance(1),
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

            AL_middle_train_ = np.concatenate((X_train_, AL_embedding_middle_features), axis=1)
            L2_middle_train_ = np.concatenate((X_train_, L2_embedding_middle_features), axis=1)

            # Build a cascade layer
            layer_idx = self.n_layers_
            layer_ = []

            AL_middle_train_ = self.buffer_.cache_data(
                layer_idx, AL_middle_train_, is_training_data=True
            )
            L2_middle_train_ = self.buffer_.cache_data(
                layer_idx, L2_middle_train_, is_training_data=True
            )

            if self.verbose > 0:
                msg = "{} Fitting cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            tic = time.time()
            ALSH = ALSHForest(100, 4)
            ALSH.fit(L2_middle_train_)
            layer_.append(ALSH)
            L2SH = L2SHForest(100, 4)
            L2SH.fit(AL_middle_train_)
            layer_.append(L2SH)
            toc = time.time()
            training_time = toc - tic

            if self.verbose > 0:
                msg = "{} layer = {:<2} | {} | Elapsed = {:.3f} s"
                print(
                    msg.format(
                        _utils.ctime(),
                        layer_idx,
                        self._repr_performance(1),
                        training_time
                    )
                )

            # Update the cascade layer
            self._set_layer(layer_idx, layer_)
            self.n_layers_ += 1

        self.is_fitted_ = True

        return self

    def clean(self):
        """
        Clean the buffer created by the model if ``partial_mode`` is ``True``.
        """
        if self.partial_mode:
            self.buffer_.close()


## Implementation of the deep forest for classification.

class CascadeLSHForest(BaseCascadeLSHForest):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _repr_performance(self, pivot):
        msg = "Val Acc = {:.3f} %"
        return msg.format(pivot * 100)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        proba : :obj:`numpy.ndarray` of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if not self.is_fitted_:
            raise AttributeError("Please fit the model first.")
        self._check_input(X)

        if self.verbose > 0:
            print("{} Start to evalute the model:".format(_utils.ctime()))

        X_test = X
        #X_middle_test_ = _utils.init_array(X_test, self.n_aug_features_)

        for layer_idx in range(self.n_layers_):
            layer = self._get_layer(layer_idx)
            if self.verbose > 0:
                msg = "{} Evaluating cascade layer = {:<2}"
                print(msg.format(_utils.ctime(), layer_idx))

            if layer_idx == 0:
                #X_aug_test_ = layer.transform(X_test)
                if self.embedding_features:
                    AL_middle_features = layer[0].decision_path(X_test)
                    L2_middle_features = layer[1].decision_path(X_test)
            elif layer_idx < self.n_layers_-1:
                AL_middle_test_ = np.concatenate((X_test, AL_middle_features), axis=1)
                L2_middle_test_ = np.concatenate((X_test, L2_middle_features), axis=1)
                AL_embedding_middle_features = layer[0].decision_path(L2_middle_test_)
                L2_embedding_middle_features = layer[1].decision_path(AL_middle_test_)

            else:
                AL_middle_test_ = np.concatenate((X_test, AL_embedding_middle_features), axis=1)
                L2_middle_test_ = np.concatenate((X_test, L2_embedding_middle_features), axis=1)

        if self.verbose > 0:
            msg = "{} Final evaluating layer = {:<2}"
            print(msg.format(_utils.ctime(), self.n_layers_-1))
        layer = self._get_layer(self.n_layers_-1)
        AL_proba = layer[0].decision_function(L2_middle_test_)
        L2_proba = layer[1].decision_function(AL_middle_test_)
        proba = np.concatenate((AL_proba, L2_proba), axis=1)
        proba = _utils.merge_proba(proba, self.n_outputs_)

        return proba

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``np.uint8``.

        Returns
        -------
        y : :obj:`numpy.ndarray` of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        a = np.array(proba)
        #print(a[:,0])

        return a[:,0]