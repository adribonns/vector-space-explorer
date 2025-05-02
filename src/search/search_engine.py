"""
Search Module - Finds nearest neighbors of a new vector in the existing space.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_nearest(vector, vectors, top_k=5):
    """
    Finds the top_k closest vectors based on cosine similarity.

    :param vector: Query vector.
    :type vector: np.ndarray
    :param vectors: List of stored vectors.
    :type vectors: np.ndarray
    :param top_k: Number of nearest vectors to return.
    :type top_k: int
    :return: Indices of closest vectors.
    :rtype: list[int]
    """
    similarities = cosine_similarity([vector], vectors)[0]
    return np.argsort(similarities)[::-1][:top_k]
