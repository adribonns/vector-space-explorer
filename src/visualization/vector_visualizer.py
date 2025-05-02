"""
Visualization Module - 2D and 3D visualizations using PCA, UMAP, TSNE.
"""

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

def pca_3d(vectors, labels=None):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors)
    df = pd.DataFrame(reduced, columns=["x", "y", "z"])
    if labels is not None:
        df["label"] = labels
    return px.scatter_3d(df, x="x", y="y", z="z", color="label" if labels is not None else None)

def tsne_2d(vectors, labels=None):
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vectors)
    df = pd.DataFrame(reduced, columns=["x", "y"])
    if labels is not None:
        df["label"] = labels
    return px.scatter(df, x="x", y="y", color="label" if labels is not None else None)

def umap_2d(vectors, labels=None):
    reducer = umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(vectors)
    df = pd.DataFrame(reduced, columns=["x", "y"])
    if labels is not None:
        df["label"] = labels
    return px.scatter(df, x="x", y="y", color="label" if labels is not None else None)
