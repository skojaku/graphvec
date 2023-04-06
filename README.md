# A Python package for graph embedding methods 
[![Unit Test & Deploy](https://github.com/skojaku/graphvec/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/graphvec/actions/workflows/main.yml)

This package contains algorithms for graph embedding for networks.
All algorithms are implemented in Python, with speed accelerations by numba.

# Contents
- [Installation](#installation)
- [Examples](#examples)
- [Available Algorithms](#available-algorithms)

# Installation

```bash
pip install git+https://github.com/skojaku/graphvec.git
```

For `conda` users,

```bash
git clone https://github.com/skojaku/graphvec
cd graphvec
conda develop .
```

### Dependency:

See requirements.txt

# Usage

Following the scikit-learn's API design, all algorithms have two methods`.fit` and `.transform`. The `.fit` method takes a network to learn the network structure, and the `.transform` produces the emebdding of nodes. More specifically, follow the steps below. 

First, load a graph embedding method. For demonstration, we use `Node2Vec`. See the "Available algorithms" Section for other graph embedding methods.

```python
import graphvec
model = graphvec.Node2Vec()
```

Second, call `.fit` with the network `A`:
```python
model.fit(A)
```
where `A` is the adjacency matrix in scipy.sparse format. For networkx user, networkx.Graph can be converted to the scipy.sparse format by `A = nx.adjacency_matrix(G)`.

Lastly, call `.transform` to generate an embedding:
```python
emb = model.transform(dim=64) 
```
where `dim` is the number of dimensions. `emb` is a numpy.ndarray with size (number of nodes, dim). The ith row in `emb` (`emb[i, :]`) is the embedding vector of the ith node in the given adjacency matrix. 



# Available algorithms

| Algorithm | Reference |
|-----------|-----------|
| [graphvec.Node2Vec](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L47) | [Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.](https://dl.acm.org/doi/abs/10.1145/2939672.2939754?casa_token=7cmZ7FTEFQUAAAAA:kzky_dEcYk2HJvCWR6Oj0WVlQ8kXYDPzne7DH9JrNwJaVMQwqLIsR72chSmYu3gFvavK5IHz5uA)|
| [graphvec.DeepWalk](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L129)  | [Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.](https://dl.acm.org/doi/abs/10.1145/2623330.2623732?casa_token=M-iuSKzREZkAAAAA:PO_lT11Wqf7gpkbGmip_2yOPVaDoBEstKFmPWGTMQ0EQpGYCrbY5n06yFoCHoSc3MA1L876sEyE) |
| [graphvec.LaplacianEigenMap](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L140)  | [Belkin, Mikhail, and Partha Niyogi. "Laplacian eigenmaps and spectral techniques for embedding and clustering." Advances in neural information processing systems 14 (2001).](https://proceedings.neurips.cc/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html) |
| [graphvec.ModularityMatrixSpectralEmbedding](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L237)  | [Nadakuditi, Raj Rao, and Mark EJ Newman. "Graph spectra and the detectability of community structure in networks." Physical review letters 108.18 (2012): 188701.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.188701)|
| [graphvec.AdjacencyMatrixSpectralEmbedding](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L205)  | The spectrum of the adjacency matrix |
| [graphvec.NonBacktrackingSpectralEmbedding](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L237)  | [Krzakala, Florent, et al. "Spectral redemption in clustering sparse networks." Proceedings of the National Academy of Sciences 110.52 (2013): 20935-20940.](https://www.pnas.org/doi/abs/10.1073/pnas.1312486110) |
| [graphvec.FastRP](https://github.com/skojaku/graphvec/blob/617c3a9ab3b5a859c1957507144ae6853871b602/graphvec/embeddings.py#L291) | [Chen, Haochen, et al. "Fast and accurate network embeddings via very sparse random projection." Proceedings of the 28th ACM international conference on information and knowledge management. 2019.](https://dl.acm.org/doi/abs/10.1145/3357384.3357879) |

## For development

### Install 
```bash 
conda create -n graphvec python=3.9
conda activate graphvec
conda install -c conda-forge mamba -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y
mamba install -c conda-forge numpy==1.23.5
```
