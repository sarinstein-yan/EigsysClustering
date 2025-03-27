import torch
import numpy as np
from typing import Tuple, Union
from numpy.typing import ArrayLike

#-------------------------------
# JIT-compiled helper functions
#-------------------------------

@torch.jit.script
def pairwise_sq_dists(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Compute the squared Euclidean distance between each row of x and each row of y.
    x: (n, d), y: (m, d)
    Returns: (n, m) matrix where entry (i, j) = ||x_i - y_j||^2.
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).t()
    dists = x_norm + y_norm - 2 * (x @ y.t())
    return torch.clamp(dists, min=0.0)

@torch.jit.script
def default_h(x: torch.Tensor) -> torch.Tensor:
    """
    Default isotropic kernel function h(x) = exp(-x).
    """
    return torch.exp(-x)

@torch.jit.script
def compute_kernel_sym_matrix(
    pts: torch.Tensor, 
    epsilon: float, 
    diffusion_alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the symmetric kernel matrix M, along with the intermediate degree vectors.
    - pts: data tensor of shape (N, d)
    - epsilon: kernel bandwidth (proximity parameter).
    - diffusion_alpha: diffusion parameter.
    
    Returns:
      M: the symmetrically normalized kernel matrix,
      q: row sum of the isotropic kernel,
      deg: row sum of the renormalized (anisotropic) kernel.
    """
    # Compute the raw kernel matrix using the default kernel function
    K = default_h(pairwise_sq_dists(pts, pts) / epsilon)
    # First degree: row sum of the isotropic kernel
    q = K.sum(dim=1)
    # Normalize by q (Coifman’s normalization)
    K = K / ((q.unsqueeze(1) * q.unsqueeze(0)) ** diffusion_alpha)
    # Second degree: row sum of the renormalized kernel
    deg = K.sum(dim=1)
    # Symmetric normalization
    M = K / torch.sqrt(deg.unsqueeze(1) * deg.unsqueeze(0))
    return M, q, deg

#-------------------------------
# DiffusionMapTorch class following scikit-learn API
#-------------------------------

class DiffusionMapTorch:
    def __init__(self, 
            diffusion_alpha: float = 1.0, 
            epsilon: float = 0.1, 
            diffusion_time: float = 1.0, 
            n_components: int = None, 
            kernel_function=None
        ):
        """
        Spectral embedding using diffusion maps.

        Parameters
        ----------
        diffusion_alpha : float, default=1.0
            Diffusion parameter (typically in [0,1]) controlling the normalization of the kernel.
            - When alpha=1, the kernel is normalized by the row sum of the isotropic kernel, removing
            the influence of the sampling density entirely.
            - When alpha=0, use the raw kernel matrix without normalization; the resulting diffusion 
            process is influenced by the data's sampling density.
        epsilon : float, default=0.1
            Kernel bandwidth (proximity parameter) used in the construction of the kernel matrix.
        diffusion_time : float, default=1.0
            Diffusion time parameter used to scale the eigenvalues (λ^t).
        n_components : int, optional
            Number of spectral embedding coordinates to retain. If None, all components are kept.
        kernel_function : callable, optional
            Kernel function to use. Must be compatible with torch.jit.script. If None, the default 
            h(x)=exp(-x) is used.
        """
        self.diffusion_alpha = diffusion_alpha
        self.epsilon = epsilon
        self.diffusion_time = diffusion_time
        self.n_components = n_components
        self.kernel_function = kernel_function if kernel_function is not None else default_h

        # Attributes to be set during fit.
        self.X_ = None               # Training data, shape (N, d)
        self.embedding_ = None       # Diffusion map (spectral embedding) of the training data, shape (N, n_components)
        self.eigenvalues_ = None     # Leading eigenvalues from the kernel matrix.
        self.eigenvectors_ = None    # Corresponding eigenvectors.
        self.row_sum_ = None         # Row sum of the isotropic kernel.
        self.degree_ = None          # Row sum of the renormalized (anisotropic) kernel.
        self.kernel_matrix_ = None   # Symmetric normalized kernel matrix.

    def fit(self, X: Union[ArrayLike, torch.Tensor], y=None):
        """
        Fit the diffusion map model on the training data X, and compute the spectral embedding.

        Parameters
        ----------
        X : array-like or torch.Tensor, shape (N, d)
            Training data where each row corresponds to an observation.
        y : None
            Ignored; included for compatibility with scikit-learn.

        Returns
        -------
        self : DiffusionMapTorch
            The fitted estimator.
        """
        if not isinstance(X, torch.Tensor):
            try:
                X = torch.tensor(X)
            except Exception as e:
                raise ValueError("Input data must be a tensor or convertible to a tensor.") from e
        
        self.X_ = X
        N = X.shape[0]
        if self.n_components is None:
            self.n_components = N

        # Compute the symmetric kernel matrix and degree vectors.
        self.kernel_matrix_, self.row_sum_, self.degree_ = compute_kernel_sym_matrix(self.X_, self.epsilon, self.diffusion_alpha)

        # Eigen decomposition of the symmetric kernel matrix.
        eigenvals, eigenvecs = torch.linalg.eigh(self.kernel_matrix_)
        # torch.linalg.eigh returns eigenvalues in ascending order.
        idx = torch.argsort(eigenvals, descending=True)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        self.eigenvalues_ = eigenvals[:self.n_components]
        self.eigenvectors_ = eigenvecs[:, :self.n_components]

        # Convert eigenvectors to diffusion map (spectral embedding) coordinates.
        deg_sqrt = torch.sqrt(self.degree_ + 1e-10)  # add a tiny constant for numerical stability.
        # Normalize eigenvectors by the square root of the anisotropic degree.
        psi = self.eigenvectors_ / deg_sqrt.unsqueeze(1)
        # Scale by eigenvalues raised to the diffusion time.
        self.embedding_ = psi * (self.eigenvalues_ ** self.diffusion_time)
        return self

    def transform(self, X: Union[ArrayLike, torch.Tensor]) -> torch.Tensor:
        """
        Apply the learned diffusion map (spectral embedding) to new data X using the Nystrom extension.

        Parameters
        ----------
        X : array-like or torch.Tensor, shape (M, d)
            New data points to be embedded.

        Returns
        -------
        X_new : torch.Tensor, shape (M, n_components)
            Spectral embedding coordinates for the new data.
        """
        if self.embedding_ is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")

        if not isinstance(X, torch.Tensor):
            try:
                X = torch.tensor(X)
            except Exception as e:
                raise ValueError("Input data must be a tensor or convertible to a tensor.") from e

        P = self._compute_transition_probabilities(X)
        select = torch.arange(0, self.n_components)
        # Compute the scaled training embedding (Ψ divided by eigenvalues).
        PsiLambda = self.embedding_[:, select] / (self.eigenvalues_[select].unsqueeze(0) + 1e-10)
        X_new = P @ PsiLambda
        return X_new

    def fit_transform(self, X: Union[ArrayLike, torch.Tensor], y=None) -> torch.Tensor:
        """
        Fit the model with X and return the spectral embedding for the training data.

        Parameters
        ----------
        X : array-like or torch.Tensor, shape (N, d)
            Training data.
        y : None
            Ignored; included for compatibility with scikit-learn.

        Returns
        -------
        X_new : torch.Tensor, shape (N, n_components)
            Spectral embedding coordinates for the training data.
        """
        self.fit(X, y)
        return self.embedding_

    def _compute_transition_probabilities(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the transition probability matrix between new data points X and the training data.

        Parameters
        ----------
        X : torch.Tensor, shape (M, d)
            New data points.

        Returns
        -------
        P : torch.Tensor, shape (M, N)
            Transition probability matrix from new points to the training data.
        """
        dists = pairwise_sq_dists(X, self.X_)
        y = -dists / self.epsilon - self.diffusion_alpha * torch.log(self.row_sum_ + 1e-10)
        y = y - y.max(dim=1, keepdim=True).values
        exp_y = torch.exp(y)
        P = exp_y / exp_y.sum(dim=1, keepdim=True)
        return P

    def compute_transition_matrix(self) -> torch.Tensor:
        """
        Compute the transition probability matrix for the training data.

        Returns
        -------
        P : torch.Tensor, shape (N, N)
            Transition probability matrix computed using the learned kernel.
        """
        dists = pairwise_sq_dists(self.X_, self.X_)
        y = -dists / self.epsilon - self.diffusion_alpha * torch.log(self.row_sum_ + 1e-10)
        y = y - y.max(dim=1, keepdim=True).values
        exp_y = torch.exp(y)
        P = exp_y / exp_y.sum(dim=1, keepdim=True)
        return P

    def nystrom_extension(self, X: torch.Tensor, return_P: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extend the spectral embedding to new data points using the Nystrom method.

        Parameters
        ----------
        X : torch.Tensor, shape (M, d)
            New data points for which to compute the embedding.
        return_P : bool, default False
            If True, also return the transition probability matrix used in the extension.

        Returns
        -------
        extension : torch.Tensor, shape (M, n_components)
            Spectral embedding coordinates for the new data.
        P : torch.Tensor, shape (M, N), optional
            Transition probability matrix if return_P is True.
        """
        if self.embedding_ is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")

        P = self._compute_transition_probabilities(X)
        select = torch.arange(0, self.n_components)
        PsiLambda = self.embedding_[:, select] / (self.eigenvalues_[select].unsqueeze(0) + 1e-10)
        extension = P @ PsiLambda
        if return_P:
            return extension, P
        return extension

    def nystrom_on_training_data(self, save_P: bool = False):
        """
        Compute the Nystrom extension for the training data (self.X_) itself.
        Optionally save the computed transition probability matrix.

        Parameters
        ----------
        save_P : bool, default False
            If True, saves the computed transition probability matrix as attribute 'P_nyst_'.
        """
        if self.embedding_ is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")
        result = self.nystrom_extension(self.X_, return_P=save_P)
        if save_P:
            self.embedding_nyst_, self.P_nyst_ = result
        else:
            self.embedding_nyst_ = result

    def compute_out_of_sample_probabilities(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the simplified transition probability matrix for new points X using the kernel function.

        Parameters
        ----------
        X : torch.Tensor, shape (M, d)
            New data points.

        Returns
        -------
        P : torch.Tensor, shape (M, N)
            Transition probability matrix representing the probabilities for each new point.
        """
        Kzx = self.kernel_function(pairwise_sq_dists(X, self.X_) / self.epsilon)
        # Adjust by the row sum (with exponent diffusion_alpha)
        Kzx = Kzx / (self.row_sum_.unsqueeze(0) ** self.diffusion_alpha + 1e-10)
        Z = Kzx.sum(dim=1, keepdim=True)
        return Kzx / (Z + 1e-10)