# graphvec

My toolkit for graph embedding.


## Install
```
pip install https://github.com/skojaku/graphvec
```

## A minimal example 

```python 
import graphvec

model = graphvec.Node2Vec(dim = 64) # dim is the number of embedding dimension 
model.fit(A) # A is the adjacency mtarix in form of the scipy sparse matrix
emb = model.transform() # emb is a (number of nodes, dim) numpy array, with each row corresponding to an embedding. 
```

## Implemented methods 

- `Node2Vec`
- `DeepWalk`
- `AdjacencyMatrixSpectralEmbedding`
- `LaplacianEigenMap`
- `ModularityMatrixSpectralEmbedding`
- `NonBacktrackingSpectralEmbedding`
- `FastRP`

See the codes for the parameters


## Development

### Install 
```bash 
conda create -n graphvec python=3.9
conda activate graphvec
conda install -c conda-forge mamba -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y
mamba install -c conda-forge numpy==1.23.5
```
