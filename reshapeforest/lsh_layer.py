"""Implementation of the forest-based cascade layer."""


__all__ = ["Layer"]

import numpy as np

from . import _utils
from .lsh_estimator import Estimator


def _build_estimator(
    X,
    layer_idx,
    estimator_idx,
    estimator_name,
    estimator,
    oob_decision_function,
    partial_mode=True,
    buffer=None,
    verbose=1
):
    """Private function used to fit a single estimator."""
    if verbose > 1:
        msg = "{} - Fitting estimator = {:<5} in layer = {}"
        key = estimator_name + "_" + str(estimator_idx)
        print(msg.format(_utils.ctime(), key, layer_idx))

    estimator.fit_transform(X)

    if partial_mode:
        # Cache the fitted estimator in out-of-core mode
        buffer_path = buffer.cache_estimator(
            layer_idx, estimator_idx, estimator_name, estimator
        )
        return buffer_path   #X_aug_train,
    else:
        return estimator    #X_aug_train,


class Layer(object):

    def __init__(
        self,
        layer_idx,
        n_classes=1,
        n_estimators=3, #modify it according to the real estimators
        n_trees=100,
        granularity=1,
        partial_mode=False,
        buffer=None,
        n_jobs=4,
        verbose=1,
    ):
        self.layer_idx = layer_idx
        self.n_classes = n_classes
        self.n_estimators = n_estimators   # internal conversion
        self.n_trees = n_trees
        self.real_estimator = n_estimators

        self.granularity = granularity,

        self.partial_mode = partial_mode
        self.buffer = buffer
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Internal container
        self.estimators_ = {}

    @property
    def n_trees_(self):
        return self.n_estimators * self.n_trees

    def _make_estimator(self, estimator_idx, estimator_name):
        """Make and configure a copy of the estimator."""

        estimator = Estimator(
            name=estimator_name,
            n_trees=self.n_trees,
            #granularity=self.granularity,
            n_jobs=None
        )

        return estimator

    def _validate_params(self):

        if not self.n_estimators > 0:
            msg = "`n_estimators` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_estimators))

        if not self.n_trees > 0:
            msg = "`n_trees` = {} should be strictly positive."
            raise ValueError(msg.format(self.n_trees))

    def fit_transform(self, X):

        self._validate_params()
        n_samples, _ = X.shape

        oob_decision_function = np.zeros((n_samples, self.n_classes))

        # A random forest and an extremely random forest will be fitted
        for estimator_idx in range(self.n_estimators // self.real_estimator):
            _estimator = _build_estimator(
                X,
                self.layer_idx,
                estimator_idx,
                "ALSH",
                self._make_estimator(estimator_idx, "ALSH"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
            )
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "ALSH")
            self.estimators_.update({key: _estimator})

        for estimator_idx in range(self.n_estimators // self.real_estimator):
            _estimator = _build_estimator(
                X,
                self.layer_idx,
                estimator_idx,
                "L1SH",
                self._make_estimator(estimator_idx, "L1SH"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
            )
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "L1SH")
            self.estimators_.update({key: _estimator})

        for estimator_idx in range(self.n_estimators // self.real_estimator):
            _estimator = _build_estimator(
                X,
                self.layer_idx,
                estimator_idx,
                "L2SH",
                self._make_estimator(estimator_idx, "L2SH"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
            )
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "L2SH")
            self.estimators_.update({key: _estimator})

        # Set the OOB estimations and validation accuracy
        #self.oob_decision_function_ = oob_decision_function / self.n_estimators
        # y_pred = np.argmax(oob_decision_function, axis=1)
        # self.val_acc_ = accuracy_score(y, y_pred)

        # X_aug = np.hstack(X_aug)
        # return X_aug

    def transform(self, X):
        """
        Return the concatenated transformation results from all base
        estimators."""
        n_samples, _ = X.shape
        X_aug = np.zeros((n_samples, self.n_classes * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split('-')[-1] + "_" + str(key.split('-')[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_classes*idx, self.n_classes*(idx+1)
            X_aug[:, left:right] += estimator.transform(X)

        return X_aug

    def predict_full(self, X):
        """Return the concatenated predictions from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_classes * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split('-')[-1] + "_" + str(key.split('-')[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_classes*idx, self.n_classes*(idx+1)
            pred[:, left:right] += estimator.predict(X)

        return pred

    def path_train(self, X):
        """Return the concatenated path node features from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_classes * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split('-')[-1] + "_" + str(key.split('-')[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_classes * idx, self.n_classes * (idx + 1)
            pred[:, left:right] += estimator.train_path(X)

        return pred

    def path_test(self, X):
        """Return the concatenated path node features from all base estimators."""
        n_samples, _ = X.shape
        pred = np.zeros((n_samples, self.n_classes * self.n_estimators))
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key = key.split('-')[-1] + "_" + str(key.split('-')[-2])
                print(msg.format(_utils.ctime(), key, self.layer_idx))
            if self.partial_mode:
                # Load the estimator from the buffer
                estimator = self.buffer.load_estimator(estimator)

            left, right = self.n_classes * idx, self.n_classes * (idx + 1)
            pred[:, left:right] += estimator.path_traverse(X)

        return pred
