__all__ = ["Estimator"]

from .lsh_forest import ALSHForest, L1SHForest, L2SHForest

def make_estimator(
    name,
    n_trees=100,
    n_jobs=4
):
    # ALSHForest
    if name == "ALSH":
        estimator = ALSHForest(
            num_trees=n_trees,
            n_jobs=n_jobs
        )
    # L1SHForest
    elif name == "L1SH":
        estimator = L1SHForest(
            num_trees=n_trees,
            n_jobs=n_jobs
        )
    # L2SHForest
    elif name == "L2SH":
        estimator = L2SHForest(
            num_trees=n_trees,
            n_jobs=n_jobs
        )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):

    def __init__(
        self,
        name,
        n_trees=100,
        n_jobs=4
    ):
        self.percentage_removal = 0.95
        self.estimator_ = make_estimator(name,
                                         n_trees,
                                         n_jobs)


    def fit_transform(self, X):

        self.estimator_.fit(X)

    def transform(self, X):

        return self.estimator_.decision_function(X)

    def predict(self, X):

        return self.estimator_.decision_function(X)

    def train_path(self, X):

        return self.estimator_.train_path(X)

    def path_traverse(self, X):

        return self.estimator_.decision_path(X)


