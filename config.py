"""
Configuration for Smart Document Chat API

Adjust these settings when moving between machines or changing models.
"""

# --- LLM Settings ---
OLLAMA_MODEL = "mistral:7b"
OLLAMA_URL = "http://localhost:11434"
TEMPERATURE = 0.3
TOP_P = 0.9

# --- Retrieval Settings ---
N_RESULTS_DEFAULT = 3       # Number of chunks to retrieve per query
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# --- Chunking Settings ---
CHUNK_SIZE = 800            # Max characters per chunk
CHUNK_OVERLAP = 200         # Overlap between consecutive chunks

# --- Storage Settings ---
CHROMA_COLLECTION = "documents"
CHROMA_PERSIST_DIR = "./chroma_db"
DOCUMENTS_FOLDER = "./documents"
