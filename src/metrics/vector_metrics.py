"""
Metrics Module - Compute characteristics of the vector space.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_sparsity(vectors):
    """
    Computes the average sparsity of the vector space.

    :param vectors: Array of vectors.
    :type vectors: np.ndarray
    :return: Sparsity score (0 to 1).
    :rtype: float
    """
    total_elements = vectors.size
    zero_elements = np.sum(vectors == 0)
    return zero_elements / total_elements

def compute_norms(vectors):
    """
    Computes L2 norms of the vectors.

    :param vectors: Array of vectors.
    :type vectors: np.ndarray
    :return: List of norms.
    :rtype: list[float]
    """
    return np.linalg.norm(vectors, axis=1).tolist()

def average_norm(vectors):
    """
    Computes the average L2 norm across all vectors.

    :param vectors: Array of vectors.
    :type vectors: np.ndarray
    :return: Average norm.
    :rtype: float
    """
    return float(np.mean(np.linalg.norm(vectors, axis=1)))

def mean_cosine_similarity(vectors):
    """
    Computes the mean cosine similarity across all vector pairs.

    :param vectors: Array of vectors.
    :type vectors: np.ndarray
    :return: Mean cosine similarity.
    :rtype: float
    """
    similarity_matrix = cosine_similarity(vectors)
    np.fill_diagonal(similarity_matrix, 0)
    return float(np.sum(similarity_matrix) / (vectors.shape[0] * (vectors.shape[0] - 1)))

def variance_per_dimension(vectors):
    """
    Computes the variance across each dimension of the vector space.

    :param vectors: Array of vectors.
    :type vectors: np.ndarray
    :return: Array of variances.
    :rtype: np.ndarray
    """
    return np.var(vectors, axis=0)

def vector_space_dimension(vectors):
    """
    Returns the dimension of the vector space.

    :param vectors: Array of vectors.
    :type vectors: np.ndarray
    :return: Dimensionality of vector space.
    :rtype: int
    """
    return vectors.shape[1]
