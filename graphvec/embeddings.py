# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-08-26 09:51:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-05 21:56:03
"""Module for embedding."""
# %%
import gensim
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csgraph
from scipy.optimize import curve_fit
from graphvec import samplers, utils


#
# Base class
#
class NodeEmbeddings:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass


class Node2Vec(NodeEmbeddings):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        window_length=10,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.sampler = samplers.Node2VecWalkSampler(
            num_walks=num_walks,
            walk_length=walk_length,
            p=p,
            q=q,
        )

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": 1,
            "workers": 4,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.window_length

        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(
            sentences=self.sampler.walks, **self.w2vparams
        )

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv[i]
            self.out_vec[i, :] = self.model.syn1neg[self.model.wv.key_to_index[i]]


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams = {
            "sg": 1,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }


class LaplacianEigenMap(NodeEmbeddings):
    def __init__(self, p=100, q=40, reconstruction_vector=False):
        self.in_vec = None
        self.L = None
        self.deg = None
        self.p = p
        self.q = q
        self.reconstruction_vector = reconstruction_vector

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        if self.reconstruction_vector:
            s, u = sparse.linalg.eigs(self.L, k=dim, which="LR")
            s, u = np.real(s), np.real(u)
            order = np.argsort(s)[::-1]
            self.in_vec = u[:, order]
            self.out_vec = u[:, order]
        else:
            s, u = sparse.linalg.eigs(self.L, k=dim + 1, which="LR")
            s, u = np.real(s), np.real(u)
            order = np.argsort(-s)[1:]
            s, u = s[order], u[:, order]
            Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
            self.in_vec = Dsqrt @ u @ sparse.diags(np.sqrt(np.abs(s)))
            self.out_vec = u


class AdjacencyMatrixSpectralEmbedding(NodeEmbeddings):
    def __init__(
        self,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):
        svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
        u = svd.fit_transform(self.A)
        s = svd.singular_values_
        self.in_vec = u @ sparse.diags(s)


class ModularityMatrixSpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, reconstruction_vector=False, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.reconstruction_vector = reconstruction_vector
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):

        self.A = self.A.asfptype()
        s, u = sparse.linalg.eigs(self.A, k=dim + 1, which="LR")
        s, u = np.real(s), np.real(u)
        s = s[1:]
        u = u[:, 1:]

        if self.reconstruction_vector:
            is_positive = s > 0
            u[:, ~is_positive] = 0
            s[~is_positive] = 0
            self.in_vec = u @ sparse.diags(np.sqrt(s))
        else:
            self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = u


class NonBacktrackingSpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, auto_dim=False):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.auto_dim = auto_dim
        self.C = 10

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.A = A
        return self

    def update_embedding(self, dim):

        N = self.A.shape[0]
        Z = sparse.csr_matrix((N, N))
        I = sparse.identity(N, format="csr")
        D = sparse.diags(self.deg)
        B = sparse.bmat([[Z, D - I], [-I, self.A]], format="csr")

        if self.auto_dim is False:
            s, v = sparse.linalg.eigs(B, k=dim, tol=1e-4)
            s, v = np.real(s), np.real(v)
            order = np.argsort(-np.abs(s))
            s, v = s[order], v[:, order]
            v = v[N:, :]

            # Normalize the eigenvectors because we cut half the vec
            # and omit the imaginary part.
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

            # Weight the dimension by the eigenvalues
            v = v @ np.diag(np.sqrt(np.abs(s)))
        else:
            dim = int(self.C * np.sqrt(N))
            dim = np.minimum(dim, N - 1)

            s, v = sparse.linalg.eigs(B, k=dim + 1, tol=1e-4)

            c = int(self.A.sum() / N)
            s, v = s[np.abs(s) > c], v[:, np.abs(s) > c]

            order = np.argsort(s)
            s, v = s[order], v[:, order]
            s, v = s[1:], v[:, 1:]
            v = v[N:, :]
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

        self.in_vec = v


class FastRP(NodeEmbeddings):
    def __init__(self, window_size=10, s=3.0, beta=-1):
        """Fast Random Projection embedding

        See https://arxiv.org/abs/1908.11512.

        Examples:

        Getting an embedding of a network
        >> emb = fastRP(net, dim=256, window_size=10)

        If the network is directed, one can construct two types of an embedding, one based on the neighbors
        connected by in-coming edges, and those by out-going edges. You can get the two embeddings by setting `edge_direction=True`.

        >> emb_out, emb_in = fastRP(net, dim=256, window_size=10, edge_direction=True)

        where `emb_out` is an embedding of nodes based on the edges going from the nodes, and `emb_in` is based on the in-coming edges.

        :param net: Adjacency matrix
        :type net: scipy.sparse matrix
        :param dim: Embedding dimension
        :type dim: int
        :param window_size: Window size
        :type window_size: int
        :param beta: degree normalization, defaults to -1. Ranges [0,1]. beta = -1 is the strongest normalization. beta = 0 means no regularization.
        :type beta: int, optional
        :param s: Inverse density of the random matrix, defaults to 3.0
        :type s: float, optional
        :return: embedding matrix
        :rtype: numpy.ndarray of (number of nodes, dimension)
        """
        self.window_size = window_size
        self.s = s
        self.beta = beta
        self.out_vec = None
        self.in_vec = None

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):

        self.in_vec, self.out_vec = self.fastRP(
            self.A,
            dim,
            window_size=self.window_size,
            beta=self.beta,
            s=self.s,
            edge_direction=True,
        )

    def fastRP(self, net, dim, window_size, beta=-1, s=3.0, edge_direction=False):

        n_nodes = net.shape[0]
        # Generate random matrix for random projection
        X = sparse.random(
            n_nodes,
            dim,
            density=1 / s,
            data_rvs=lambda x: np.sqrt(s) * (2 * np.random.randint(2, size=x) - 1),
        ).toarray()

        emb = self._fastRP(net, dim, window_size, beta=beta, X=X.copy())

        if edge_direction:
            emb_cnt = self._fastRP(
                net=sparse.csr_matrix(net.T),
                dim=dim,
                window_size=window_size,
                beta=beta,
                X=X.copy(),
            )
            return emb, emb_cnt
        return emb

    def _fastRP(self, net, dim, window_size, X, beta=-1):

        # Get stats
        n_nodes = net.shape[0]
        outdeg = np.array(net.sum(axis=1)).reshape(-1)
        indeg = np.array(net.sum(axis=0)).reshape(-1)

        # Construct the transition matrix
        P = sparse.diags(1 / np.maximum(1, outdeg)) @ net  # Transition matrix
        L = sparse.diags(np.power(np.maximum(indeg.astype(float), 1.0), beta))

        # First random projection
        X0 = (P @ L) @ X.copy()  # to include the self-loops

        # h is an array for normalization
        h = np.ones((n_nodes, 1))
        h0 = h.copy()

        # Iterative projection
        for _ in range(window_size):
            X = P @ X + X0
            h = P @ h + h0

        # Normalization
        X = sparse.diags(1.0 / np.maximum(np.array(h).reshape(-1), 1e-8)) @ X
        return X


class LINE(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, window_length=1, **params)

class SpectralGraphTransformation(NodeEmbeddings):
    """
    Spectral graph transformation

    Fits and transforms input graph using spectral graph transformation method.

    Parameters
    ----------
    NodeEmbeddings : _type_
        _description_

    Attributes
    ----------
    kernel_matrix : str
        String indicating kernel matrix to use (options: "A", "normalized_A", "laplacian")
    kernel_func : callable
        Function defining the kernel function used in the transformation
    in_vec, out_vec : ndarray or None
        Input and output node embeddings

    Methods
    -------
    fit(net)
        Fits the model on the input graph
    update_embedding(dim)
        Transforms the fitted graph into low-dimensional space
    get_kernel_matrix(A)
        Computes the kernel matrix from an adjacency matrix
    train_test_edge_split(A, fraction)
        Splits edge set into training and testing sets

    References
    ----------
    - https://dl.acm.org/doi/abs/10.1145/1553374.1553447
    """

    def __init__(self, kernel_func="exp", kernel_matrix="normalized_A"):
        """
        Initialize SpectralGraphTransformation with given kernel function and
        kernel matrix options.

        Parameters
        ----------
        kernel_func : str or callable, optional (default="exp")
            The kernel function used for the spectral graph transformation.
            Can be a string representing one of two built-in kernel functions ("exp" or "neu"),
            or a user-defined function taking two input values as follows:
            `kernel_func(x, a) -> y`, where x is an array of eigenvalues, a is a scalar parameter,
            and y is the transformed array of eigenvalues.
        kernel_matrix : str, optional (default="A")
            The kernel matrix used in the transformation. Can be one of three options:
            "A" for adjacency matrix, "normalized_A" for normalized adjacency matrix,
            or "laplacian" for Laplacian matrix.
        """
        self.kernel_matrix = kernel_matrix

        if kernel_func == "exp":
            self.kernel_func = lambda x, a: np.exp(-a * x)
        elif kernel_func == "neu":
            self.kernel_func = lambda x, a: 1.0 / (1 - a * x)
        else:
            self.kernel_func = kernel_func
        self.in_vec = None
        self.out_vec = None

    def fit(self, net):
        """
        Fit the spectral graph transformation model on the input graph.

        Parameters
        ----------
        net : array-like or sparse matrix
            Input graph as an adjacency matrix or equivalent sparse matrix format.

        Returns
        -------
        self : object
            Returns self.
        """

        A = utils.to_adjacency_matrix(net)
        n_nodes = A.shape[0]
        train_edges, test_edges = self.train_test_edge_split(A, 1 / 3)
        train_net = utils.edge2network(train_edges[0], train_edges[1], n_nodes=n_nodes)
        test_net = utils.edge2network(test_edges[0], test_edges[1], n_nodes=n_nodes)

        self.A = A
        self.train_net = train_net
        self.test_net = test_net
        self.Gkernel = self.get_kernel_matrix(A)
        self.Gkernel_train = self.get_kernel_matrix(train_net)

        return self

    def update_embedding(self, dim):
        """
        Update the input network embedding and transform it into low-dimensional space.

        Parameters
        ----------
        dim : int
            Dimensions of the output embedding.

        Returns
        -------
        None
        """

        which = "LR"
        if self.kernel_matrix == "laplacian":
            which = "SR"

        s_train, u = sparse.linalg.eigs(self.Gkernel_train, k=dim, which=which)
        s_test = np.diag(u.T @ self.Gkernel @ u)
        s_test, u, s_train = np.real(s_test), np.real(u), np.real(s_train)

        popt, pcov = curve_fit(
            self.kernel_func,
            s_train / np.max(np.abs(s_train)),
            s_test / np.max(np.abs(s_test)),
            p0=[-5e-1],
        )
        alpha = popt[0]
        spred = self.kernel_func(s_test, alpha)
        self.in_vec = u @ np.diag(np.sqrt(np.abs(spred)))
        self.out_vec = self.in_vec

    def get_kernel_matrix(self, A):
        deg = np.array(A.sum(axis=1)).reshape(-1)
        if self.kernel_matrix == "A":
            M = A
        elif self.kernel_matrix == "normalized_A":
            M = sparse.diags(1 / np.sqrt(deg)) @ A @ sparse.diags(1 / np.sqrt(deg))
        elif self.kernel_matrix == "laplacian":
            M = sparse.diags(deg) - A
        return sparse.csr_matrix(M)

    def train_test_edge_split(self, A, fraction):
        r, c, _ = sparse.find(A)
        edges = np.unique(utils.pairing(r, c))

        MST = csgraph.minimum_spanning_tree(A + A.T)
        r, c, _ = sparse.find(MST)
        mst_edges = np.unique(utils.pairing(r, c))
        remained_edge_set = np.array(
            list(set(list(edges)).difference(set(list(mst_edges))))
        )
        max_fraction = len(remained_edge_set) / len(edges)
        if fraction > max_fraction:
            fraction = max_fraction * 0.5
        n_edge_removal = int(fraction * len(remained_edge_set))
        test_edge_set = np.random.choice(
            remained_edge_set, n_edge_removal, replace=False
        )

        train_edge_set = np.array(
            list(set(list(edges)).difference(set(list(test_edge_set))))
        )

        test_edges_ = utils.depairing(test_edge_set)
        train_edges_ = utils.depairing(train_edge_set)
        return train_edges_, test_edges_
