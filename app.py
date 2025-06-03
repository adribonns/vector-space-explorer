import streamlit as st
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.metrics.vector_metrics import compute_sparsity, average_norm, mean_cosine_similarity, vector_space_dimension, variance_per_dimension
from src.visualization.vector_visualizer import pca_visualize, tsne_visualize, umap_visualize
from data.input_sentences import INPUT_SENTENCES

st.set_page_config(layout="wide")
st.title("🚀 Vector Space Explorer")

sentences = INPUT_SENTENCES
# Embedding
model_name = st.sidebar.selectbox("Choose embedding model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
model = SentenceTransformer(model_name)
embeddings = model.encode(sentences, convert_to_numpy=True)

st.subheader("📊 Metrics")
col1, col2 = st.columns(2)
col1.metric("Sparsity", f"{compute_sparsity(embeddings):.4f}")
col2.metric("Average L2 Norm", f"{average_norm(embeddings):.4f}")
col3, col4 = st.columns(2)
col3.metric("Vector Space Dimension", f"{vector_space_dimension(embeddings)}")
col4.metric("Mean Cosine Similarity", f"{mean_cosine_similarity(embeddings):.4f}")

with st.expander("Show variance per dimension"):
    st.write(variance_per_dimension(embeddings))

st.subheader("📈 Visualization")
method = st.sidebar.selectbox("Select the reduction method", ["PCA", "t-SNE", "UMAP"])
dim = st.sidebar.radio("Select the reduction dimension", [2, 3])

if method == "PCA":
    fig = pca_visualize(embeddings, sentences, dim=dim)
elif method == "t-SNE":
    fig = tsne_visualize(embeddings, sentences, dim=dim)
elif method == "UMAP":
    fig = umap_visualize(embeddings, sentences, dim=dim)

st.plotly_chart(fig, use_container_width=True)
