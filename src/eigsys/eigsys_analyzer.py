import numpy as np

def eigsys_vecs(eigvecs, eigvals, is_Hermitian=True, tol=1e-10):
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
    eigvecs : array_like, shape [..., N, N]
    eigvals : array_like, shape [..., N]
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