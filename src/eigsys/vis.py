import numpy as np

from sklearn.decomposition import PCA
# from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels

def distance_matrix(X, metric='euclidean', sort=False):
    dist = pairwise_distances(X, metric=metric)
    if sort:
        emb = PCA(n_components=1).fit_transform(X).ravel()
        idx = np.argsort(emb)
        dist = dist[idx][:, idx]
    return dist

def kernel_matrix(X, metric='linear', sort=False):
    ker = pairwise_kernels(X, metric=metric)
    if sort:
        emb = PCA(n_components=1).fit_transform(X).ravel()
        idx = np.argsort(emb)
        ker = ker[idx][:, idx]
    return ker