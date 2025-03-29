import numpy as np

def eigsys_vecs(eigvals, eigvecs, is_Hermitian=True, tol=1e-10):
    """
    Process eigen-systems and return a concatenated vector per system.
    
    Hermitian:
        Output [eigvals.real, flattened |eigvecs|].
    Non-Hermitian:
        For each eigenvector (column), gauge fix so its largest-magnitude
        component is real and positive; then output
        [eigvals.real, eigvals.imag, flattened |gauge-fixed eigvecs|,
         flattened phase of gauge-fixed eigvecs].
    
    Parameters
    ----------
    eigvals : array_like, shape [..., N]
    eigvecs : array_like, shape [..., N, N]
    is_Hermitian : bool, optional
        If True, no gauge fixing is done (default True).
    tol : float, optional
        Threshold below which values are set to zero.
    
    Returns
    -------
    out : ndarray
        Array of shape (system_dims..., D) where D = N + N*N for Hermitian,
        and D = 2*N + 2*(N*N) for non-Hermitian systems.
    """
    eigvecs = np.asarray(eigvecs)
    eigvals = np.asarray(eigvals)
    
    # Get system dimensions (all but the last two) and number of eigenvectors N.
    sys_shape = eigvecs.shape[:-2]
    N = eigvecs.shape[-1]
    num_sys = int(np.prod(sys_shape)) if sys_shape else 1
    
    # Reshape to (num_sys, N, N) and (num_sys, N)
    eigvecs = eigvecs.reshape(num_sys, N, N)
    eigvals = eigvals.reshape(num_sys, N)
    
    # Chop very small values.
    eigvecs[np.abs(eigvecs) < tol] = 0.0
    
    if is_Hermitian:
        out = np.concatenate([eigvals.real, np.abs(eigvecs).reshape(num_sys, -1)], axis=1)

    else:
        # Gauge fixing: for each eigenvector (each column), find index of its max abs component.
        abs_vecs = np.abs(eigvecs)
        max_idx = np.argmax(abs_vecs, axis=1)  # shape (num_sys, N)
        # Extract max components for each eigenvector.
        max_comp = np.take_along_axis(eigvecs, max_idx[:, np.newaxis, :], axis=1).squeeze(axis=1)
        # Compute gauge factor so that the max component becomes real and positive.
        gauge = np.exp(-1j * np.angle(max_comp))[:, np.newaxis, :]
        eigvecs = eigvecs * gauge
        eigvecs[np.abs(eigvecs) < tol] = 0.0
        
        out = np.concatenate([
            eigvals.real,
            eigvals.imag,
            np.abs(eigvecs).reshape(num_sys, -1),
            np.angle(eigvecs).reshape(num_sys, -1)
        ], axis=1)
    
    return out.reshape(sys_shape + (out.shape[-1],))


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from .diffusion_clustering import DiffusionClustering
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, adjusted_mutual_info_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# helper function to add noise to a dataset
def add_noise(X, sample_noise_level=0., feature_noise_level=0.):
    """
    Add noise to the dataset X.
    Noise level is measured by the standard deviation of feature (column-wise)
    and / or sample (row-wise). E.g., level = 2 means add Gaussian noise
    with std = 2 * std.
    
    Parameters:
    - X: Input dataset (numpy array).
    - sample_noise_level: Noise level to be added to each sample (row-wise).
    - feature_noise_level: Noise level to be added to each feature (column-wise).
    
    Returns:
    - X_noise: Dataset with added noise.
    """

    sample_std = np.std(X, axis=1, keepdims=True)
    feature_std = np.std(X, axis=0, keepdims=True)
    X_noise = X.copy()
    if sample_noise_level > 0:
        noise_sample = sample_noise_level * np.random.randn(*X.shape) * sample_std
        X_noise += noise_sample
    if feature_noise_level > 0:
        noise_feature = feature_noise_level * np.random.randn(*X.shape) * feature_std
        X_noise += noise_feature
    return X_noise

class ClusteringEvaluator:
    def __init__(self, 
                 n_clusters=2, 
                 random_state=0, 
                 methods=None,
                 diffusion_time=1, 
                 epsilon=0.1,
        ):
        """
        Cluster evaluator following scikit-learn’s estimator interface.
        
        Parameters:
            n_clusters (int): Number of clusters.
            random_state (int): Random seed.
            methods (iterable): Optional iterable specifying the clustering methods to use.
                Each element can be:
                    - A string alias.
                        valid options: 
                        - 'gmm': Gaussian Mixture Model
                        - 'kmeans': KMeans
                        - 'spectral': Spectral Clustering (Nearest Neighbors affinity)
                        - 'dmp': Diffusion Map (Gaussian Kernel)
                    - A callable with a scikit-learn–style interface (i.e. it accepts X and returns (model, y_pred)).
                        Example: 
                        ```python
                        def custom_clustering(X):
                            # For demonstration, we'll simply run KMeans
                            model = KMeans(n_clusters=2, random_state=0)
                            model.fit(X)
                            y_pred = model.labels_
                            return model, y_pred
                        ```
                    The corresponding alias will be the function name. In this case, 'custom_clustering'.

            diffusion_time (int): Diffusion time for DiffusionClustering. Useless if DiffusionMap is not used.
            epsilon (float): Epsilon parameter for DiffusionClustering. Useless if DiffusionMap is not used.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.diffusion_time = diffusion_time
        self.epsilon = epsilon
        
        # Container for fitted model instances.
        self.X = None; self.y = None
        self.fitted_models = {}
        self.result_df = None
        self.result_dict = None
        self.noise_test_result = None
        
        # Default alias mapping for built-in methods.
        self.aliases = {
            'kmeans': 'KMeans',
            'gmm': 'GaussianMixture',
            'agg': 'AgglomerativeClustering',
            'spectral': 'SpectralClustering',
            'dmp': 'DiffusionMap',
        }
        
        # If methods is provided, build the clustering_methods dictionary accordingly.
        if methods is None:
            self.clustering_methods = {
                'KMeans': self.run_kmeans,
                'GaussianMixture': self.run_gmm,
                'AgglomerativeClustering': self.run_agglomerative,
                'SpectralClustering': self.run_spectral,
                'DiffusionMap': self.run_diffusion_map,
            }
        else:
            self.clustering_methods = {}
            custom_count = 1
            # Use built-in mapping for string aliases.
            for m in methods:
                if isinstance(m, str):
                    key = self.aliases.get(m.lower())
                    if key is None:
                        raise ValueError(f"Alias '{m}' not recognized. Valid aliases: {list(self.aliases.keys())}")
                    if key == 'KMeans':
                        self.clustering_methods[key] = self.run_kmeans
                    elif key == 'GaussianMixture':
                        self.clustering_methods[key] = self.run_gmm
                    elif key == 'AgglomerativeClustering':
                        self.clustering_methods[key] = self.run_agglomerative
                    elif key == 'SpectralClustering':
                        self.clustering_methods[key] = self.run_spectral
                    elif key == 'DiffusionMap':
                        self.clustering_methods[key] = self.run_diffusion_map
                elif callable(m):
                    # For custom callables, use its __name__ if available, else assign a custom name.
                    name = getattr(m, '__name__', None)
                    if name is None:
                        name = f"custom_method_{custom_count}"
                        custom_count += 1
                    self.clustering_methods[name] = m
                else:
                    raise ValueError("Each method must be either an alias string or a callable.")
    
        # Define evaluation metrics.
        self.metrics = {
            'Accuracy': lambda y, y_pred, X: self.safe_accuracy(y, y_pred),
            'ARI': lambda y, y_pred, X: adjusted_rand_score(y, y_pred),
            'AMI': lambda y, y_pred, X: adjusted_mutual_info_score(y, y_pred),
            'Silhouette': lambda y, y_pred, X: self.safe_metric(silhouette_score, X, y_pred),
            'CalinskiHarabasz': lambda y, y_pred, X: self.safe_metric(calinski_harabasz_score, X, y_pred),
            'DaviesBouldin': lambda y, y_pred, X: self.safe_metric(davies_bouldin_score, X, y_pred),
        }
    
    def run_kmeans(self, X):
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        model.fit(X)
        y_pred = model.labels_
        return model, y_pred
    
    def run_gmm(self, X):
        model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        model.fit(X)
        y_pred = model.predict(X)
        return model, y_pred
    
    def run_agglomerative(self, X):
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        y_pred = model.fit_predict(X)
        return model, y_pred
    
    def run_spectral(self, X):
        model = SpectralClustering(n_clusters=self.n_clusters, random_state=self.random_state, 
                                   affinity='nearest_neighbors')
        y_pred = model.fit_predict(X)
        return model, y_pred
    
    def run_diffusion_map(self, X):
        model = DiffusionClustering(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            diffusion_time=self.diffusion_time, 
            epsilon=self.epsilon
        )
        y_pred = model.fit_predict(X)
        return model, y_pred
    
    @staticmethod
    def safe_metric(metric_fn, X_data, labels):
        """
        Compute an unsupervised metric only if at least two clusters are found.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return np.nan
        return metric_fn(X_data, labels)
    
    @staticmethod
    def safe_accuracy(y_true, y_pred):
        """
        Computes accuracy for binary classification, ensuring label alignment.
        """
        if np.unique(y_true).size != 2 or np.unique(y_pred).size != 2:
            print("Accuracy is only defined for binary classification (i.e. only two phases)."
                  " Returning NaN.")
            return np.nan
        if y_true[0] != y_pred[0]:
            y_pred = 1 - y_pred
        return accuracy_score(y_true, y_pred)
    
    def fit(self, X, y):
        """
        Runs the clustering methods and computes evaluation metrics.
        
        Parameters:
            X (np.array or pd.DataFrame): Data of shape [n_samples, n_features].
            y (np.array): True labels of shape [n_samples].
            
        Returns:
            self: The fitted instance.
        """
        self.X = X; self.y = y

        results = []
        self.fitted_models = {}
        
        for method_name, cluster_fn in self.clustering_methods.items():
            try:
                model, y_pred = cluster_fn(X)
                self.fitted_models[method_name] = model
            except Exception as e:
                print(f"Method {method_name} failed with error: {e}")
                y_pred = None
            
            if y_pred is not None:
                for metric_name, metric_fn in self.metrics.items():
                    try:
                        score = metric_fn(y, y_pred, X)
                    except Exception as e:
                        score = np.nan
                        print(f"Metric {metric_name} for {method_name} failed: {e}")
                    results.append({
                        'Method': method_name,
                        'Metric': metric_name,
                        'Score': score
                    })
        
        self.result_df = pd.DataFrame(results)
        
        # Build a dictionary mapping each method to its evaluation scores.
        self.result_dict = {}
        for method in self.result_df['Method'].unique():
            df_method = self.result_df[self.result_df['Method'] == method]
            self.result_dict[method] = dict(zip(df_method['Metric'], df_method['Score']))
        
        return self
    
    def get_model(self, alias_or_aliases):
        """
        Retrieves fitted model(s) by alias.
        
        Parameters:
            alias_or_aliases (str or list/tuple of str): Alias(es) for the clustering method(s).
                For built-in methods, supported aliases include:
                    'kmeans', 'gmm', 'agg', 'spectral', 'dmp'.
                Custom callables can be retrieved by their __name__.
        
        Returns:
            A single fitted model if a single alias is provided,
            or a tuple of fitted models if multiple aliases are provided.
            
        Raises:
            ValueError if an alias is not recognized or the model has not been fitted.
        """
        if isinstance(alias_or_aliases, str):
            alias_or_aliases = [alias_or_aliases]
        
        models = []
        for alias in alias_or_aliases:
            # First, check if alias is directly in the fitted models.
            if alias in self.fitted_models:
                models.append(self.fitted_models[alias])
            else:
                # Try mapping with built-in aliases.
                mapped = self.aliases.get(alias.lower())
                if mapped is not None and mapped in self.fitted_models:
                    models.append(self.fitted_models[mapped])
                else:
                    raise ValueError(f"Model for alias '{alias}' has not been fitted yet.")
        return models[0] if len(models) == 1 else tuple(models)
    
    def as_spreadsheet(self):
        """
        Returns the evaluation results as a pivoted spreadsheet-like table.
        
        The pivoted table has clustering methods as rows, evaluation metrics as columns,
        and the corresponding scores as values. Rows are sorted by 'Accuracy' in descending order.
        
        Returns:
            pd.DataFrame: Pivoted evaluation table.
            
        Raises:
            ValueError: If the model has not been fitted.
        """
        if self.result_df is None:
            raise ValueError("The model has not been fitted yet. Please call fit() first.")
        pivot_df = self.result_df.pivot(index='Method', columns='Metric', values='Score')
        if 'Accuracy' in pivot_df.columns:
            pivot_df = pivot_df.sort_values('Accuracy', ascending=False)
        return pivot_df
    
    def to_wideform(self):
        """
        Returns the evaluation results as a pivoted spreadsheet-like table.
        Alternative name for `as_spreadsheet()`.
        
        Returns:
            pd.DataFrame: Pivoted evaluation table.
        """
        return self.as_spreadsheet()

    def noise_robustness_test(self, 
            noise_levels, 
            method='gmm', 
            X=None, y=None, 
            n_iter=10,
        ):
        """
        For each noise level in noise_levels, adds feature noise to X n_iter times,
        fits the specified clustering method on the noisy data, and computes metrics: Accuracy, ARI, and AMI.
        
        Parameters:
            noise_levels (iterable): Iterable of noise levels to test.
            X (np.array or pd.DataFrame): Data of shape [n_samples, n_features].
                If None, uses the data from the last fit.
            y (np.array): True labels of shape [n_samples].
                If None, uses the labels from the last fit.
            n_iter (int): Number of iterations per noise level.
            method (str or callable): Clustering method to test. For built-in methods, use alias strings 
                such as 'gmm', 'kmeans', 'spectral', 'dmp'. For custom methods, provide the callable.
                Default is 'gmm'.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['Noise Level', 'Accuracy', 'ARI', 'AMI'] containing the test results.
        
        The test result is stored in the class attribute `noise_test_result`.
        """
        X = self.X if X is None else X
        y = self.y if y is None else y

        from tqdm import tqdm
        results = []

        # Determine the clustering method function to use.
        if isinstance(method, str):
            mapped = self.aliases.get(method.lower())
            if mapped is None or mapped not in self.clustering_methods:
                raise ValueError(f"Method alias '{method}' not recognized or not available.")
            method_fn = self.clustering_methods[mapped]
        elif callable(method):
            name = getattr(method, '__name__', None)
            if name is None or name not in self.clustering_methods:
                raise ValueError("Provided custom clustering method is not available in clustering_methods.")
            method_fn = self.clustering_methods[name]
        else:
            raise ValueError("Method must be a string alias or a callable.")

        for level in noise_levels:
            for i in tqdm(range(n_iter), desc=f"Noise Level {level}", total=n_iter):
                X_noisy = add_noise(X, feature_noise_level=level)
                try:
                    model, y_pred = method_fn(X_noisy)
                except Exception as e:
                    print(f"Method {method} failed at noise level {level} iteration {i}: {e}")
                    continue
                acc = self.safe_accuracy(y, y_pred)
                ari = adjusted_rand_score(y, y_pred)
                ami = adjusted_mutual_info_score(y, y_pred)
                results.append([level, acc, ari, ami])
        
        df_results = pd.DataFrame(results, columns=['Noise Level', 'Accuracy', 'ARI', 'AMI'])
        self.noise_test_result = df_results
        return df_results