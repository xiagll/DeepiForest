"""
Implementation of the forest model for classification in Deep Forest.

"""

__all__ = ["ALSHForest","L1SHForest","L2SHForest"]

import threading
import math
from .tree import LSHTree
from .sampling import VSSampling
from .tree import AngleLSH, E2LSH, KernelLSH
from joblib import Parallel, delayed
from joblib import effective_n_jobs
from sklearn.decomposition import PCA

from sklearn.utils.fixes import _joblib_parallel_args
from scipy.sparse import hstack as sparse_hstack

import numpy as np
import copy as cp


def _parallel_build_trees(
        lsh_instance,
        data
):
    """
    Private function used to fit a single tree in parallel."""

    # Fit the tree on the bootstrapped samples
    tree = LSHTree(lsh_instance)
    tree.fit(data)

    return tree

def _accumulate_prediction(_tree, data, granularity, out, lock):
    """This is a utility function for joblib's Parallel."""

    depths = []
    data_size = len(data)
    for i in range(data_size):
        transformed_data = data[i]
        d_depth=_tree.predict(granularity, transformed_data)
        depths.append(d_depth)
    depths = np.array(depths)

    with lock:
        if len(out) == 1:
            out[0] += depths
        else:
            for i in range(len(out)):
                out[i] += depths[i]

def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

class LSHForest:
    def __init__(self, num_trees, lsh_family, n_jobs=None, verbose=0, granularity=1):
        self._num_trees = num_trees
        self._lsh_family = lsh_family
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._granularity = granularity
        self._sampler = VSSampling(self._num_trees)
        self.percentage_removal = 0.6
        self._trees = []
        self._pca = None
        self._pca_input = None

    def display(self):
        for t in self._trees:
            t.display()

    def fit(self, data):
        self.build(data)

    def build(self, data):
        #X = data
        indices = range(len(data))
        # Uncomment the following code for continuous values
        data = np.c_[indices, data]

        # Important: clean the tree array
        self._trees = []

        # Sampling data
        self._sampler.fit(data)
        sampled_datas = self._sampler.draw_samples(data)

        # Build LSH instances based on the given data
        lsh_instances = []
        for i in range(self._num_trees):
            transformed_data = data
            self._lsh_family.fit(transformed_data)
            lsh_instances.append(cp.deepcopy(self._lsh_family))


        # Build LSH trees in parallel
        n_jobs, _, _ = _partition_estimators(self._num_trees, self.n_jobs)
        #lock = threading.Lock()
        rets = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                        **_joblib_parallel_args(prefer='threads',
                                                require="sharedmem"))(
            delayed(_parallel_build_trees)(
                t,
                sampled_datas[i])
            for i, t in enumerate(lsh_instances))

        # Collect newly grown trees
        for tree in rets:
            self._trees.append(tree)


    def train_path(self,data):
        print(self._num_trees)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              **_joblib_parallel_args(prefer="threads"))(
            delayed(self._trees[i].decision_path)(data)
            for i in range(self._num_trees))
        out = sparse_hstack(indicators).tocsr()
        summed_paths = np.sum(out, axis=0)
        selected_indexes = np.where(summed_paths / data.shape[0] < self.percentage_removal)
        selected_summed_nodes = summed_paths[selected_indexes]
        selected_nodes = out[:, selected_indexes[1]]
        #selected_nodes = selected_nodes.toarray()
        # weights = (1 / (np.log(selected_summed_nodes + math.e)))
        # pca_input = np.multiply(selected_nodes, weights)
        #
        # temp = np.sum(pca_input, axis=1)
        # sum_node = (temp / self._num_trees * 0.1).A
        weights = 1 / (np.log(selected_summed_nodes + math.e))
        temp = selected_nodes * weights.T
        # pca_input = selected_nodes * weights
        # x = selected_nodes.shape[0]
        # temp = []
        # for i in range(x):
        #     c = np.dot(selected_nodes[i], weights[0])
        #     temp.append(c)

        # temp = np.sum(pca_input, axis=1)
        sum_node = (temp / self._num_trees * 0.1).reshape(len(data),1)

        #return self._pca.transform(pca_input)
        return sum_node

    def decision_path(self, data):
        """
        Return the decision path in the forest.
        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.
        """
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              **_joblib_parallel_args(prefer="threads"))(
            delayed(self._trees[i].decision_path)(data)
            for i in range(self._num_trees))
        out = sparse_hstack(indicators).tocsr()
        summed_paths = np.sum(out, axis=0)
        selected_indexes = np.where(summed_paths / data.shape[0] < self.percentage_removal)
        selected_summed_nodes = summed_paths[selected_indexes]
        selected_nodes = out[:, selected_indexes[1]]
        #selected_nodes = selected_nodes.toarray()
        # weights = (1 / (np.log(selected_summed_nodes + math.e)))
        # pca_input = np.multiply(selected_nodes, weights)
        #
        # temp = np.sum(pca_input, axis=1)
        # sum_node = (temp / self._num_trees * 0.1).A
        weights = 1 / (np.log(selected_summed_nodes + math.e))
        temp = selected_nodes * weights.T
        #pca_input = selected_nodes * weights
        # x = selected_nodes.shape[0]
        # temp = []
        # for i in range(x):
        #     c = np.dot(selected_nodes[i], weights[0])
        #     temp.append(c)

        # temp = np.sum(pca_input, axis=1)
        sum_node = (temp / self._num_trees * 0.1).reshape(len(data),1)

        #return self._pca.transform(pca_input)
        return sum_node

    def decision_function(self, data):
        indices = range(len(data))
        data = np.c_[indices, data]

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self._num_trees, self.n_jobs)

        # Avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros(data.shape[0], dtype=np.float64)]
        all_proba = np.array(all_proba)

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(
                self._trees[i],
                data,
                self._granularity,
                all_proba,
                lock)
            for i in range(self._num_trees))

        # Arithmatic mean
        for proba in all_proba:
            proba /= self._num_trees
        if len(all_proba) == 1:
            all_proba = all_proba[0]
        else:
            all_proba = all_proba
        all_probas = np.array(all_proba).reshape(len(data),1)

        return all_probas


    def get_avg_branch_factor(self):
        sum = 0.0
        for t in self._trees:
            sum += t.get_avg_branch_factor()
        return sum / self._num_trees


class ALSHForest(LSHForest):
    def __init__(self, num_trees, n_jobs=None, verbose=0, granularity=1):
        super().__init__(
            num_trees=num_trees,
            lsh_family=AngleLSH(),
            n_jobs=n_jobs,
            verbose=verbose,
            granularity=granularity
        )
        # self._lsh_family = AngleLSH()
        self._sampler = VSSampling(self._num_trees)
        self._trees = []

class L1SHForest(LSHForest):
    def __init__(self, num_trees, n_jobs=None, verbose=0, granularity=1):
        super().__init__(num_trees=num_trees,
            lsh_family=E2LSH(norm=1),
            n_jobs=n_jobs,
            verbose=verbose,
            granularity=granularity)
        self._sampler = VSSampling(self._num_trees)
        self._trees = []

class L2SHForest(LSHForest):
    def __init__(self, num_trees, n_jobs=None, verbose=0, granularity=1):
        super().__init__(num_trees=num_trees,
            lsh_family=E2LSH(norm=2),
            n_jobs=n_jobs,
            verbose=verbose,
            granularity=granularity)
        self._sampler = VSSampling(self._num_trees)
        self._trees = []
