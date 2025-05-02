"""
ChromaDB Manager Module - Handles importing and preprocessing of text documents into ChromaDB.
"""

import chromadb
from sentence_transformers import SentenceTransformer

class ChromaManager:
    """
    A class to manage ChromaDB collections and handle text preprocessing.
    """

    def __init__(self, collection_name: str = "text_embeddings"):
        """
        Initializes the ChromaManager with a specified collection name.

        :param collection_name: Name of the ChromaDB collection.
        :type collection_name: str
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, documents: list[str]):
        """
        Embeds and adds documents to the ChromaDB collection.

        :param documents: List of text documents to embed.
        :type documents: list[str]
        """
        vectors = self.model.encode(documents).tolist()
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            self.collection.add(documents=[doc], embeddings=[vector], ids=[f"id_{i}"])
