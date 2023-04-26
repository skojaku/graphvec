# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-26 12:02:42
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-26 18:01:57
# %%
import numpy as np
from scipy import sparse
from scipy.special import expit
from graphvec import samplers, utils
from numba import njit
import random
from tqdm import tqdm


class NumpyNode2Vec:
    """
    Class to perform node embedding using the Node2Vec algorithm with numpy.

    Args:
        n_nodes (int): Number of nodes in the graph.
        dim (int): Dimension of the embedding vector.
        window_length (int, optional): Length of the context window. Defaults to 10.
        num_walks (int, optional): Number of walks per node. Defaults to 40.
        walk_length (int, optional): Length of each walk. Defaults to 80.
        window_type (str, optional): Type of window, can be 'double', 'left', or 'right'. Defaults to 'double'.
        **params: Additional parameters used by the ADAM optimizer.

    Attributes:
        invec (ndarray): Input embedding matrix.
        outvec (ndarray): Output embedding matrix.
        num_walks (int): Number of walks per node.
        walk_length (int): Length of each walk.
        window_length (int): Length of the context window.
        window_type (int): Type of window, can be -1 for left, 0 for double or 1 for right.
        bias (ndarray): Bias term used in the embedding calculation.
        adam_invec (ADAM): Optimizer used for input embedding.
        adam_outvec (ADAM): Optimizer used for output embedding.
        adam_bias (ADAM): Optimizer used for bias term.
    """

    def __init__(
        self,
        n_nodes,
        dim,
        window_length=10,
        num_walks=10,
        walk_length=80,
        window_type="double",
        p=1,
        q=1,
        batch_size=1,
        **params,
    ):
        self.invec = (np.random.rand(n_nodes, dim) - 0.5) / dim
        self.outvec = (np.random.rand(n_nodes, dim) - 0.5) / dim
        self.bias = np.array([0])
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_length = window_length
        self.batch_size = batch_size
        self.q = q
        self.p = p
        self.window_type = {"double": 0, "left": -1, "right": 1}[window_type]

        self.adam_invec = ADAM(self.invec.shape)
        self.adam_outvec = ADAM(self.outvec.shape)
        self.adam_bias = ADAM(self.bias.shape)

        self._rw_net = None
        self._is_weighted = False

    def train(self, net):
        self.n_nodes = net.shape[0]
        indeg = np.array(net.sum(axis=0)).reshape(-1)
        p0 = indeg / np.sum(indeg)  # negative sampling probability
        self._fit_random_walk_sampler(net)

        if self._is_weighted:
            rw_sampler = lambda start: samplers._random_walk_weighted(
                self._rw_net.indptr,
                self._rw_net.indices,
                self._rw_net.data,
                self.walk_length,
                self.p,
                self.q,
                start,
            )
        else:
            rw_sampler = lambda start: samplers._random_walk(
                self._rw_net.indptr,
                self._rw_net.indices,
                self.walk_length,
                self.p,
                self.q,
                start,
            )

        n_iter = int(self.num_walks * self.n_nodes / self.batch_size)
        pbar = tqdm(range(n_iter))
        for it in pbar:
            walks = np.array(
                [
                    rw_sampler(np.random.randint(0, self.n_nodes))
                    for _ in range(self.batch_size)
                ]
            )
            center, context = _get_center_context(
                self.window_type,
                walks=walks,
                n_walks=walks.shape[0],
                walk_len=self.walk_length,
                window_length=self.window_length,
                padding_id=-1,
            )
            random_context = np.random.choice(self.n_nodes, p=p0, size=len(context))

            # Count duplicates:
            # pairing
            pos_center_context, pos_freq = np.unique(
                _pairing(center, context), return_counts=True
            )
            neg_center_context, neg_freq = np.unique(
                _pairing(center, random_context), return_counts=True
            )
            pos_center, pos_context = _depairing(pos_center_context)
            neg_center, neg_context = _depairing(neg_center_context)
            center, context, y, freq = (
                np.concatenate([pos_center, neg_center]),
                np.concatenate([pos_context, neg_context]),
                np.concatenate([np.ones_like(pos_center), np.zeros_like(neg_center)]),
                np.concatenate([pos_freq, neg_freq]),
            )
            error = self._update(center, context, y, freq)

            pbar.set_description(f"Error={error:.3f}")

    def _fit_random_walk_sampler(self, net):
        if net.getformat() != "csr":
            raise TypeError("A should be in the scipy.sparse.csc_matrix")

        is_weighted = np.max(net.data) != np.min(net.data)
        net.sort_indices()
        if is_weighted:
            data = net.data / net.sum(axis=1).A1.repeat(np.diff(net.indptr))
            net.data = samplers._csr_row_cumsum(net.indptr, data)
        self._rw_net = net
        self._is_weighted = self._is_weighted

    def _update(self, center, context, y, freq):
        # Prediction
        score = (
            np.array(
                np.sum(self.invec[center, :] * self.outvec[context, :], axis=1)
            ).reshape(-1)
            + self.bias[0]
        )
        ypred = expit(score)

        # Prediction error
        error = freq * (y - ypred) / np.sum(freq)

        # Calculate the raw gradient
        grad_center = np.einsum("i,ij->ij", error, self.outvec[context, :])
        grad_context = np.einsum("i,ij->ij", error, self.invec[center, :])
        grad_bias = np.sum(error)

        # Sum the gradient over center
        grad_center, center = _bincount2d(grad_center, center)
        grad_context, context = _bincount2d(grad_context, context)

        # Gradient
        grad_center = self.adam_invec.update(grad=grad_center, indices=center)
        grad_context = self.adam_outvec.update(grad=grad_context, indices=context)
        grad_bias = self.adam_bias.update(grad=grad_bias, indices=np.array([0]))

        self.invec[center] += grad_center
        self.outvec[context] += grad_context
        self.bias = self.bias + grad_bias

        return np.abs(np.sum(error))


def _bincount2d(a, indices):
    indices, _idx = np.unique(indices, return_inverse=True)
    _sum = np.zeros((len(indices), a.shape[1]))
    np.add.at(_sum, _idx, a)
    return _sum, indices


def _get_center_context(
    context_window_type, walks, n_walks, walk_len, window_length, padding_id
):
    """
    Extracts center and context nodes from random walks.

    Args:
        context_window_type (int): type of the context window: 0 for double context window,
            -1 for left-side context window, and 1 for right-side context window.
        walks (np.ndarray): random walks in which to extract center and context nodes.
        n_walks (int): number of random walks.
        walk_len (int): length of each random walk.
        window_length (int): length of the context window.
        padding_id (int): id used for padding.

    Returns:
        tuple: A tuple containing three elements:
            center (np.ndarray): an array of center node ids.
            context (np.ndarray): an array of context node ids.
            freq (np.ndarray): frequency of each (center, context) pair.
    """
    if context_window_type == 0:
        center, context = _get_center_double_context_windows(
            walks, n_walks, walk_len, window_length, padding_id
        )
    elif context_window_type == -1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
        )
    elif context_window_type == 1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=False
        )
    else:
        raise ValueError("Unknown window type")
    center = np.outer(center, np.ones(context.shape[1]))
    center, context = center.reshape(-1), context.reshape(-1)
    s = (center != padding_id) * (context != padding_id)
    center, context = center[s], context[s]
    return center, context


def _pairing(r, c):
    return r + 1j * c


def _depairing(v):
    return np.real(v).astype(int), np.imag(v).astype(int)


@njit(nogil=True)
def _get_center_left_right_nodes(walks, n_walks, walk_len, window_length, padding_id):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    lefts = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    rights = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]
        for i in range(window_length):
            if t_walk - 1 - i < 0:
                break
            lefts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]

        for i in range(window_length):
            if t_walk + 1 + i >= walk_len:
                break
            rights[start:end, i] = walks[:, t_walk + 1 + i]

    return centers, lefts, rights


@njit(nogil=True)
def _get_center_single_context_window(
    walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        if is_left_window:
            for i in range(window_length):
                if t_walk - 1 - i < 0:
                    break
                contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]
        else:
            for i in range(window_length):
                if t_walk + 1 + i >= walk_len:
                    break
                contexts[start:end, i] = walks[:, t_walk + 1 + i]
    return centers, contexts


@njit(nogil=True)
def _get_center_double_context_windows(
    walks, n_walks, walk_len, window_length, padding_id
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones(
        (n_walks * walk_len, 2 * window_length), dtype=np.int64
    )
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        for i in range(window_length):
            if t_walk - 1 - i < 0:
                break
            contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]

        for i in range(window_length):
            if t_walk + 1 + i >= walk_len:
                break
            contexts[start:end, window_length + i] = walks[:, t_walk + 1 + i]

    return centers, contexts


#
# Adam optimizer
#
class ADAM:
    """
    An optimizer that implements the Adaptive Moment Estimation (ADAM) algorithm.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to be optimized.

    Attributes
    ----------
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    eta : float
        The learning rate.
    t : int
        The number of iterations.
    mt : numpy.ndarray
        The estimated first moment vector.
    vt : numpy.ndarray
        The estimated second moment vector.
    eps : float
        A small constant to avoid division by zero.
    """

    def __init__(self, shape):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eta = 1e-2

        self.eps = 1e-8
        self.mt = np.zeros(shape)
        self.vt = np.zeros(shape)
        self.t = np.zeros(shape)

    def update(self, grad, indices):
        """
        Ascending
        """

        self.t[indices] += 1
        self.mt[indices] = self.beta1 * self.mt[indices] + (1 - self.beta1) * grad
        self.vt[indices] = self.beta2 * self.vt[indices] + (
            1 - self.beta2
        ) * np.multiply(grad, grad)

        mthat = self.mt[indices] / np.maximum(
            1 - np.power(self.beta1, self.t[indices]), 1e-32
        )
        vthat = self.vt[indices] / np.maximum(
            1 - np.power(self.beta2, self.t[indices]), 1e-32
        )

        new_grad = mthat / (np.sqrt(vthat) + self.eps)

        return self.eta * new_grad


#
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_network(func):
    def wrapper(binarize=True, symmetrize=False, k_core=None, *args, **kwargs):
        net, labels, node_table = func(*args, **kwargs)
        if symmetrize:
            net = net + net.T
            net.sort_indices()

        if k_core is not None:
            knumbers = k_core_decomposition(net)
            s = knumbers >= k_core
            net = net[s, :][:, s]
            labels = labels[s]
            node_table = node_table[s]

        _, comps = connected_components(csgraph=net, directed=False, return_labels=True)
        ucomps, freq = np.unique(comps, return_counts=True)
        s = comps == ucomps[np.argmax(freq)]
        labels = labels[s]
        net = net[s, :][:, s]
        if binarize:
            net = net + net.T
            net.data = net.data * 0 + 1
        node_table = node_table[s]
        return net, labels, node_table

    return wrapper


@load_network
def load_airport_net():
    # Node attributes
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
    )

    # Edge table
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
    )
    # net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

    net = sparse.csr_matrix(
        (
            edge_table["weight"].values,
            (edge_table["source"].values, edge_table["target"].values),
        ),
        shape=(node_table.shape[0], node_table.shape[0]),
    )

    s = ~pd.isna(node_table["region"])
    node_table = node_table[s]
    labels = node_table["region"].values
    net = net[s, :][:, s]
    return net, labels, node_table


net, labels, _ = load_airport_net()
model = NumpyNode2Vec(n_nodes=net.shape[0], dim=64, batch_size=1)
model.train(net)
# %%
emb = model.invec

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns


clf = LinearDiscriminantAnalysis(n_components=2)
xy = clf.fit_transform(emb, labels)

sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels)

# %%
net.shape
