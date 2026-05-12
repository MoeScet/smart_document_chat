"""
Vector Store Module
Handles document embeddings and similarity search using ChromaDB
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from typing import List, Dict, Tuple
import uuid
from config import EMBEDDING_MODEL, CHROMA_COLLECTION, CHROMA_PERSIST_DIR

class VectorStore:
    """
    Manages document embeddings and retrieval using ChromaDB
    """

    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Initialize the vector store with persistent storage

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory path where ChromaDB will save data
        """
        collection_name = collection_name or CHROMA_COLLECTION
        persist_directory = persist_directory or CHROMA_PERSIST_DIR

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.persist_directory = persist_directory

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Create or get collection
        # get_or_create_collection will reuse existing collection if it exists
        # This allows us to keep previously indexed documents across sessions
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        print(f"Vector store initialized with collection: {collection_name}")
        print(f"Data persisted to: {persist_directory}")
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of dictionaries with 'text' and 'metadata' keys
        """
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = 3) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Search for relevant documents

        Args:
            query: The search query
            n_results: Number of results to return

        Returns:
            Tuple of (relevant_texts, metadatas, distances)
            distances are cosine distances (lower = more similar)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Extract documents, metadata, and distances
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []

        return documents, metadatas, distances
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def clear(self) -> None:
        """
        Clear all documents from the collection
        """
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        print("Vector store cleared")
    
    def format_sources(self, metadatas: List[Dict]) -> List[str]:
        """
        Format metadata into readable source citations

        Args:
            metadatas: List of metadata dictionaries

        Returns:
            List of formatted source strings
        """
        sources = []
        for meta in metadatas:
            source = f"{meta.get('source', 'Unknown')} (Page {meta.get('page', '?')})"
            sources.append(source)
        return sources

    def get_indexed_documents(self) -> List[str]:
        """
        Get a list of all unique document filenames that have been indexed

        This method retrieves all documents from the collection and extracts
        unique source filenames. Useful for showing users what documents are
        already in the vector store.

        Returns:
            List of unique document filenames (e.g., ['manual.pdf', 'guide.pdf'])
        """
        # Get all items from the collection
        # We use get() with no filters to retrieve everything
        results = self.collection.get()

        # Extract unique source filenames from metadata
        # Each chunk has metadata with 'source' field containing the filename
        sources = set()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])

        # Return as sorted list for consistent display
        return sorted(list(sources))

    def is_document_indexed(self, filename: str) -> bool:
        """
        Check if a document with the given filename is already indexed

        This prevents duplicate processing of the same PDF file.
        Users can be notified that a document is already indexed.

        Args:
            filename: The name of the document to check (e.g., 'manual.pdf')

        Returns:
            True if document exists in the collection, False otherwise
        """
        indexed_docs = self.get_indexed_documents()
        return filename in indexed_docs

    def delete_document(self, filename: str) -> int:
        """
        Delete all chunks belonging to a specific document

        This allows users to remove documents they no longer need,
        freeing up storage space and keeping the vector store clean.

        Args:
            filename: The name of the document to delete (e.g., 'manual.pdf')

        Returns:
            Number of chunks deleted
        """
        # Get all items in the collection
        results = self.collection.get()

        # Find IDs of all chunks that belong to this document
        # We match based on the 'source' field in metadata
        ids_to_delete = []
        if results and 'metadatas' in results and 'ids' in results:
            for i, metadata in enumerate(results['metadatas']):
                if metadata.get('source') == filename:
                    ids_to_delete.append(results['ids'][i])

        # Delete the chunks if any were found
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks from {filename}")

        return len(ids_to_delete)