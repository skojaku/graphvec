#conda create -n graphvec python=3.9
#conda activate graphvec
conda install -c conda-forge mamba -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y
mamba install -c conda-forge numpy==1.23.5
