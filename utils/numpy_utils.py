"""
NumPy utility functions for data processing and visualization.

This module provides utility functions for data manipulation, interpolation,
padding, and visualization using NumPy, matplotlib, and scikit-learn.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from typing import Tuple, Any, Union


def plot_clusters(data: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot data clusters using PCA dimensionality reduction.
    
    Reduces high-dimensional data to 2D using PCA and creates a scatter plot
    with different colors for each cluster/label.
    
    Args:
        data (np.ndarray): Input data array of shape (n_samples, n_features)
        labels (np.ndarray): Labels for each data point
    """
    print(f"Plotting clusters for {len(data)} samples with {len(np.unique(labels))} unique labels")
    
    # Reshape data to 2D if needed
    data_reshaped = data.reshape(data.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    components = pca.fit_transform(data_reshaped)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    x = components[:, 0]
    y = components[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(x[mask], y[mask], label=str(label), color=colors[i], alpha=0.7)

    ax.legend()
    ax.set_xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('Data Clusters Visualization')
    plt.tight_layout()
    
    output_file = "Data_Clusters.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Cluster plot saved as {output_file}")


def interpolate_matrix(matrix: np.ndarray, new_size: int) -> np.ndarray:
    """
    Interpolate a matrix to a new size along the first dimension.
    
    Uses scipy's interp1d to resize the matrix while preserving the relationships
    between data points. Useful for normalizing sequence lengths.
    
    Args:
        matrix (np.ndarray): Input matrix of shape (original_size, n_features)
        new_size (int): Target size for the first dimension
        
    Returns:
        np.ndarray: Interpolated matrix of shape (new_size, n_features)
        
    Raises:
        ValueError: If new_size is less than 1 or matrix is empty
    """
    if new_size < 1:
        raise ValueError("new_size must be at least 1")
    
    if matrix.size == 0:
        raise ValueError("Input matrix cannot be empty")
    
    print(f"Interpolating matrix from shape {matrix.shape} to ({new_size}, {matrix.shape[1]})")
    
    interp_matrix = np.zeros((new_size, matrix.shape[1]), dtype=matrix.dtype)
    
    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        
        # Create original indices normalized to [0, 1]
        x = np.linspace(0, 1, column.size)
        
        # Create interpolation function
        f = interp1d(x, column, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Create new indices
        new_x = np.linspace(0, 1, new_size)
        
        # Interpolate
        interp_matrix[:, col] = f(new_x)
    
    return interp_matrix


def pad_sequence(sequence: np.ndarray, max_len: int, padding_value: float = 0.0) -> Union[np.ndarray, None]:
    """
    Pad a sequence to a fixed length with a specified value.
    
    Pads sequences that are shorter than max_len and truncates sequences
    that are longer. Returns None for invalid inputs.
    
    Args:
        sequence (np.ndarray): Input sequence of shape (seq_len, n_features)
        max_len (int): Target sequence length
        padding_value (float): Value to use for padding
        
    Returns:
        Union[np.ndarray, None]: Padded sequence of shape (max_len, n_features) or None if invalid
        
    Examples:
        >>> seq = np.array([[1, 2], [3, 4]])
        >>> padded = pad_sequence(seq, 4, 0)
        >>> print(padded.shape)
        (4, 2)
    """
    if sequence is None or sequence.size == 0:
        print("Warning: Empty or None sequence provided")
        return None
        
    if max_len <= 0:
        print(f"Warning: Invalid max_len: {max_len}")
        return None
    
    if len(sequence.shape) != 2:
        print(f"Warning: Expected 2D sequence, got shape {sequence.shape}")
        return None
    
    seq_len, n_features = sequence.shape
    print(f"Padding sequence from length {seq_len} to {max_len}")
    
    # Create padded array
    padded_sequence = np.full((max_len, n_features), padding_value, dtype=np.float32)
    
    # Handle sequences longer than max_len (truncate)
    if seq_len > max_len:
        print(f"Warning: Sequence length {seq_len} exceeds max_len {max_len}, truncating")
        padded_sequence[:max_len, :] = sequence[:max_len, :]
    else:
        # Copy original sequence
        padded_sequence[:seq_len, :] = sequence
        # Remaining positions already filled with padding_value
    
    return padded_sequence


def normalize_sequence(sequence: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize a sequence using specified method.
    
    Args:
        sequence (np.ndarray): Input sequence to normalize
        method (str): Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        np.ndarray: Normalized sequence
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = np.min(sequence, axis=0, keepdims=True)
        max_vals = np.max(sequence, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (sequence - min_vals) / range_vals
        
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        mean_vals = np.mean(sequence, axis=0, keepdims=True)
        std_vals = np.std(sequence, axis=0, keepdims=True)
        
        # Avoid division by zero
        std_vals[std_vals == 0] = 1
        
        normalized = (sequence - mean_vals) / std_vals
        
    elif method == 'robust':
        # Robust normalization using median and IQR
        median_vals = np.median(sequence, axis=0, keepdims=True)
        q75 = np.percentile(sequence, 75, axis=0, keepdims=True)
        q25 = np.percentile(sequence, 25, axis=0, keepdims=True)
        iqr = q75 - q25
        
        # Avoid division by zero
        iqr[iqr == 0] = 1
        
        normalized = (sequence - median_vals) / iqr
        
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    print(f"Normalized sequence using {method} method")
    return normalized.astype(np.float32)
