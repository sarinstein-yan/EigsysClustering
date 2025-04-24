import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Union
from numpy.typing import ArrayLike
from eigsys.diffusion_map import DiffusionMapTorch

class DiffusionClustering:
    """
    Diffusion Clustering implements a clustering algorithm that first computes a 
    diffusion map (spectral embedding) of the data and then clusters the resulting 
    embedding using a specified clustering algorithm. By default, KMeans is used.
    
    This approach is analogous to spectral clustering in scikit-learn, where the 
    eigen-decomposition of a graph Laplacian is computed and then clustered.
    
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    diffusion_alpha : float, default=1.0
        Diffusion parameter that controls the anisotropic normalization of the kernel.
        - When diffusion_alpha = 0, the raw kernel is used, so the diffusion operator 
          reflects the local sampling density.
        - When diffusion_alpha = 1, full density normalization is applied, and the 
          diffusion operator approximates the Laplace–Beltrami operator.
    epsilon : float, default=0.1
        Kernel bandwidth (proximity parameter) used in constructing the affinity matrix.
    diffusion_time : float, default=1.0
        Diffusion time parameter (t in λ^t) that scales the eigenvalues in the embedding.
    n_components : int, optional
        Number of diffusion map (spectral embedding) coordinates to retain. If None, 
        all components are used.
    clusterer : object, optional
        A clustering estimator following the scikit-learn API. If None, KMeans is used.
    random_state : int or None, default=None
        Random state for reproducibility (passed to KMeans if it is used as the default clusterer).
    """
    
    def __init__(self, 
                 n_clusters: int = 2, 
                 diffusion_alpha: float = 1.0, 
                 epsilon: float = 0.1, 
                 diffusion_time: float = 1.0, 
                 n_components: int = None,
                 clusterer: object = None,
                 random_state: Union[int, None] = None):
        
        self.n_clusters = n_clusters
        self.diffusion_alpha = diffusion_alpha
        self.epsilon = epsilon
        self.diffusion_time = diffusion_time
        self.n_components = n_components
        self.random_state = random_state
        
        # Default to KMeans if no other clusterer is provided.
        if clusterer is None:
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        else:
            self.clusterer = clusterer
        
        # Instantiate the diffusion map transformer.
        self.diffusion_map = DiffusionMapTorch(
            diffusion_alpha=self.diffusion_alpha,
            epsilon=self.epsilon,
            diffusion_time=self.diffusion_time,
            n_components=self.n_components
        )
        
        # Attributes to be learned
        self.embedding_ = None  # Diffusion map embedding (torch.Tensor)
        self.labels_ = None     # Cluster labels (numpy array)
        self.X_ = None          # Training data

    def fit(self, X: Union[ArrayLike, torch.Tensor], y=None):
        """
        Fit the diffusion clustering model on the data X.
        
        This method first computes the diffusion map embedding of the data X via the 
        DiffusionMapTorch transformer, and then clusters the embedding using the 
        specified clustering algorithm (default: KMeans).
        
        Parameters
        ----------
        X : array-like or torch.Tensor, shape (N, d)
            Input data where each row corresponds to an observation.
        y : None
            Ignored; included for compatibility with the scikit-learn API.
            
        Returns
        -------
        self : DiffusionClustering
            The fitted estimator.
        """
        # Save the training data
        self.X_ = X
        
        # Compute the diffusion map embedding.
        self.diffusion_map.fit(X)
        self.embedding_ = self.diffusion_map.embedding_
        
        # Convert the embedding to a NumPy array for scikit-learn's clustering methods.
        if torch.is_tensor(self.embedding_):
            embedding_np = self.embedding_.detach().cpu().numpy()
        else:
            embedding_np = np.array(self.embedding_)
        
        # Cluster the embedding.
        self.clusterer.fit(embedding_np)
        self.labels_ = self.clusterer.labels_
        
        return self

    def fit_predict(self, X: Union[ArrayLike, torch.Tensor], y=None):
        """
        Fit the model on X and return the cluster labels.
        
        Parameters
        ----------
        X : array-like or torch.Tensor, shape (N, d)
            Input data.
        y : None
            Ignored; included for compatibility with the scikit-learn API.
        
        Returns
        -------
        labels : numpy.ndarray, shape (N,)
            Cluster labels assigned to each observation.
        """
        self.fit(X, y)
        return self.labels_