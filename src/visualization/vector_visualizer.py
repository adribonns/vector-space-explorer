import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def pca_visualize(vectors, labels=None, dim=2):
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(vectors)
    cols = ["x", "y", "z"][:dim]
    df = pd.DataFrame(reduced, columns=cols)
    if labels is not None:
        df["label"] = labels
    if dim == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="label" if labels is not None else None,
                            width=900, height=700)
    else:
        fig = px.scatter(df, x="x", y="y", color="label" if labels is not None else None,
                         width=900, height=700)
    return fig

def tsne_visualize(vectors, labels=None, dim=2):
    tsne = TSNE(n_components=dim, init="random", random_state=42)
    reduced = tsne.fit_transform(vectors)
    cols = ["x", "y", "z"][:dim]
    df = pd.DataFrame(reduced, columns=cols)
    if labels is not None:
        df["label"] = labels
    if dim == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="label" if labels is not None else None,
                            width=900, height=700)
    else:
        fig = px.scatter(df, x="x", y="y", color="label" if labels is not None else None,
                         width=900, height=700)
    return fig

def umap_visualize(vectors, labels=None, dim=2):
    reducer = umap.UMAP(n_components=dim, random_state=42)
    reduced = reducer.fit_transform(vectors)
    cols = ["x", "y", "z"][:dim]
    df = pd.DataFrame(reduced, columns=cols)
    if labels is not None:
        df["label"] = labels
    if dim == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="label" if labels is not None else None,
                            width=900, height=700)
    else:
        fig = px.scatter(df, x="x", y="y", color="label" if labels is not None else None,
                         width=900, height=700)
    return fig
