__all__ = ["Buffer"]

import os
import shutil
import warnings
import tempfile
from joblib import (load, dump)


class Buffer(object):

    def __init__(self,
                 use_buffer,
                 buffer_dir=None,
                 store_est=True,
                 store_pred=True,
                 store_data=False):

        self.use_buffer = use_buffer
        self.store_est = store_est and use_buffer
        self.store_pred = store_pred and use_buffer
        self.store_data = store_data and use_buffer
        self.buffer_dir = os.getcwd() if buffer_dir is None else buffer_dir

        # Create buffer
        if self.use_buffer:
            self.buffer = tempfile.TemporaryDirectory(prefix="buffer_",
                                                      dir=self.buffer_dir)

            if store_data:
                self.data_dir_ = tempfile.mkdtemp(prefix="data_",
                                                  dir=self.buffer.name)

            if store_est or store_pred:
                self.model_dir_ = tempfile.mkdtemp(prefix="model_",
                                                   dir=self.buffer.name)
                self.pred_dir_ = os.path.join(self.model_dir_, "predictor.est")

    @property
    def name(self):
        """Return the buffer name."""
        if self.use_buffer:
            return self.buffer.name
        else:
            return None

    def cache_data(self, layer_idx, X, is_training_data=True):
        if not self.store_data:
            return X

        if is_training_data:
            cache_dir = os.path.join(self.data_dir_,
                                     "joblib_train_{}.mmap".format(layer_idx))
            # Delete
            if os.path.exists(cache_dir):
                os.unlink(cache_dir)
        else:
            cache_dir = os.path.join(self.data_dir_,
                                     "joblib_test_{}.mmap".format(layer_idx))
            # Delete
            if os.path.exists(cache_dir):
                os.unlink(cache_dir)

        # Dump and reload data in the numpy.memmap mode
        dump(X, cache_dir)
        X_mmap = load(cache_dir, mmap_mode="r+")

        return X_mmap

    def cache_estimator(self, layer_idx, est_idx, est_name, est):
        if not self.store_est:
            return est

        filename = "{}-{}-{}.est".format(layer_idx, est_idx, est_name)
        cache_dir = os.path.join(self.model_dir_, filename)
        dump(est, cache_dir)

        return cache_dir

    def cache_predictor(self, predictor):
        if not self.store_pred:
            return predictor

        dump(predictor, self.pred_dir_)

        return self.pred_dir_

    def load_estimator(self, estimator_path):
        if not os.path.exists(estimator_path):
            msg = "Missing estimator in the path: {}."
            raise FileNotFoundError(msg.format(estimator_path))

        estimator = load(estimator_path)

        return estimator

    def load_predictor(self, predictor):

        if not isinstance(predictor, str):
            return predictor

        if not os.path.exists(predictor):
            msg = "Missing predictor in the path: {}."
            raise FileNotFoundError(msg.format(predictor))

        predictor = load(predictor)

        return predictor

    def del_estimator(self, layer_idx):
        for est_name in os.listdir(self.model_dir_):
            if est_name.startswith(str(layer_idx)):
                try:
                    os.unlink(os.path.join(self.model_dir_, est_name))
                except OSError:
                    msg = ("Permission denied when deleting the dumped"
                           " estimators during the early stopping stage.")
                    warnings.warn(msg, RuntimeWarning)

    def close(self):
        try:
            self.buffer.cleanup()
        except OSError:
            msg = "Permission denied when cleaning up the local buffer."
            warnings.warn(msg, RuntimeWarning)


def model_mkdir(dirname):
    if os.path.isdir(dirname):
        msg = ("The directory to be created already exists {}.")
        raise RuntimeError(msg.format(dirname))

    os.mkdir(dirname)
    os.mkdir(os.path.join(dirname, "estimator"))


def model_saveobj(dirname, obj_type, obj, partial_mode=False):
    if not os.path.isdir(dirname):
        msg = "Cannot find the target directory: {}. Please create it first."
        raise RuntimeError(msg.format(dirname))

    if obj_type in ("param", "binner"):
        if not isinstance(obj, dict):
            msg = "{} to be saved should be in the form of dict."
            raise RuntimeError(msg.format(obj_type))
        dump(obj, os.path.join(dirname, "{}.pkl".format(obj_type)))

    elif obj_type == "layer":
        if not isinstance(obj, dict):
            msg = "The layer to be saved should be in the form of dict."
            raise RuntimeError(msg)

        est_path = os.path.join(dirname, "estimator")
        if not os.path.isdir(est_path):
            msg = "Cannot find the target directory: {}."
            raise RuntimeError(msg.format(est_path))

        if partial_mode:
            for _, layer in obj.items():
                for estimator_key, estimator in layer.estimators_.items():
                    dest = os.path.join(est_path, estimator_key + ".est")
                    shutil.move(estimator, dest)

        else:
            for _, layer in obj.items():
                for estimator_key, estimator in layer.estimators_.items():
                    dest = os.path.join(est_path, estimator_key + ".est")
                    dump(estimator, dest)
    elif obj_type == "predictor":
        pred_path = os.path.join(dirname, "estimator", "predictor.est")

        if partial_mode:
            shutil.move(obj, pred_path)
        else:
            dump(obj, pred_path)
    else:
        raise ValueError("Unknown object type: {}.".format(obj_type))


def model_loadobj(dirname, obj_type, d=None):

    if not os.path.isdir(dirname):
        msg = "Cannot find the target directory: {}."
        raise RuntimeError(msg.format(dirname))

    if obj_type in ("param", "binner"):
        obj = load(os.path.join(dirname, "{}.pkl".format(obj_type)))
        return obj
    elif obj_type == "layer":
        from ._layer import Layer  # avoid circular import

        if not isinstance(d, dict):
            msg = "Loading layers requires the dict from `param.pkl`."
            raise RuntimeError(msg)

        n_estimators = d["n_estimators"]
        n_layers = d["n_layers"]
        layers = {}

        for layer_idx in range(n_layers):

            # Build a temporary layer
            layer_ = Layer(
                layer_idx=layer_idx,
                n_classes=d["n_outputs"],
                n_estimators=d["n_estimators"],
                partial_mode=d["partial_mode"],
                buffer=d["buffer"],
                verbose=d["verbose"]
            )

            for est_type in ("rf", "erf"):
                for est_idx in range(n_estimators):
                    est_key = "{}-{}-{}".format(
                        layer_idx, est_idx, est_type
                    )
                    dest = os.path.join(
                        dirname, "estimator", est_key + ".est"
                    )

                    if not os.path.isfile(dest):
                        msg = "Missing estimator in the path: {}."
                        raise RuntimeError(msg.format(dest))

                    if d["partial_mode"]:
                        layer_.estimators_.update(
                            {est_key: os.path.abspath(dest)}
                        )
                    else:
                        est = load(dest)
                        layer_.estimators_.update({est_key: est})

            layer_key = "layer_{}".format(layer_idx)
            layers.update({layer_key: layer_})
        return layers
    elif obj_type == "predictor":

        if not isinstance(d, dict):
            msg = "Loading the predictor requires the dict from `param.pkl`."
            raise RuntimeError(msg)

        pred_path = os.path.join(dirname, "estimator", "predictor.est")

        if not os.path.isfile(pred_path):
            msg = "Missing classifier in the path: {}."
            raise RuntimeError(msg.format(pred_path))

        if d["partial_mode"]:
            return os.path.abspath(pred_path)
        else:
            clf = load(pred_path)
            return clf
    else:
        raise ValueError("Unknown object type: {}.".format(obj_type))
