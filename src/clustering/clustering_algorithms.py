"""
Clustering Module - Performs clustering using KMeans and HDBSCAN.
"""

from sklearn.cluster import KMeans
import hdbscan

def kmeans_clustering(vectors, n_clusters=5):
    """
    Applies KMeans clustering to the vector space.

    :param vectors: Vector array.
    :type vectors: np.ndarray
    :param n_clusters: Number of clusters.
    :type n_clusters: int
    :return: Cluster labels.
    :rtype: list[int]
    """
    model = KMeans(n_clusters=n_clusters)
    return model.fit_predict(vectors)

def hdbscan_clustering(vectors, min_cluster_size=5):
    """
    Applies HDBSCAN clustering to the vector space.

    :param vectors: Vector array.
    :type vectors: np.ndarray
    :param min_cluster_size: Minimum size of clusters.
    :type min_cluster_size: int
    :return: Cluster labels.
    :rtype: list[int]
    """
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(vectors)
