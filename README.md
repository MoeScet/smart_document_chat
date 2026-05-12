# Smart Document Chat API

A document chat API using RAG (Retrieval Augmented Generation) to query PDF documents. Built with FastAPI with streaming responses.

## Features

- **Streaming chat** - responses appear word by word via Server-Sent Events
- **Source evidence** - each response includes the actual text chunks used as evidence
- **Web chat interface** accessible from any browser on your network
- RESTful API for document chat functionality
- Semantic search over indexed documents using BGE embeddings
- Document management (upload, delete, list)
- Centralized configuration (`config.py`)
- Runs locally using Ollama
- Auto-generated API documentation at `/docs`

## Tech Stack

- **FastAPI**: Web framework with streaming support
- **Uvicorn**: ASGI server
- **Ollama**: Local LLM (configurable model)
- **ChromaDB**: Vector database for document embeddings
- **Sentence Transformers**: Embedding generation (`BAAI/bge-large-en-v1.5`)
- **Pydantic**: Data validation

## Prerequisites

1. Python 3.8 or higher
   ```bash
   python --version
   ```

2. Ollama installed and running
   - Download from: https://ollama.com
   - Start the server:
   ```bash
   ollama serve
   ```
   - Download a model:
   ```bash
   ollama pull mistral:7b
   ```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Edit `config.py` to match your setup:

```python
OLLAMA_MODEL = "mistral:7b"       # Your Ollama model
TEMPERATURE = 0.3                  # Lower = more factual, higher = more creative
N_RESULTS_DEFAULT = 3              # Number of chunks to retrieve per query
CHUNK_SIZE = 800                   # Characters per chunk
CHUNK_OVERLAP = 200                # Overlap between chunks
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Run the API Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access points:
- `http://localhost:8000/chat` - Chat interface (web UI)
- `http://localhost:8000/docs` - Interactive API documentation

## Usage

### Method 1: Preprocessing Script

For batch document indexing:

1. Place PDF files in `documents/` folder

2. Run preprocessing:
   ```bash
   python preprocess_documents.py
   ```

3. Start API server:
   ```bash
   python main.py
   ```

### Method 2: Web Interface

1. Start API server:
   ```bash
   python main.py
   ```

2. Open browser and navigate to:
   ```
   http://localhost:8000/chat
   ```

3. Start chatting with your documents. Responses stream in word by word with source evidence displayed below each answer.

### Method 3: API

Upload a document:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

Query via standard endpoint (full JSON response):
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main topic?", "chat_history": []}'
```

Query via streaming endpoint (Server-Sent Events):
```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main topic?", "chat_history": []}'
```

## API Endpoints

### Chat

**POST /chat** - Send a message, get full JSON response

Response:
```json
{
  "response": "This document is about...",
  "sources": [
    {
      "filename": "report.pdf",
      "page": 3,
      "text": "The actual chunk text used as evidence..."
    }
  ],
  "sources_text": "Sources:\n[report.pdf - Page 3]\n\"The actual chunk text...\""
}
```

**POST /chat/stream** - Send a message, get streaming SSE response

Each event is a JSON object:
```
data: {"type": "token", "content": "word"}
data: {"type": "sources", "sources": [...], "sources_text": "..."}
data: {"type": "done"}
```

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents/indexed` | List all indexed documents |
| GET | `/documents/stats` | Get document and chunk counts |
| POST | `/documents/upload` | Upload and index a PDF |
| DELETE | `/documents/{filename}` | Remove a document from index |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Vector store and Ollama status |

## Configuration

All settings are in `config.py`:

```python
# LLM Settings
OLLAMA_MODEL = "mistral:7b"       # Model name from `ollama list`
OLLAMA_URL = "http://localhost:11434"
TEMPERATURE = 0.3                  # 0.1-0.3 for factual, 0.5-0.7 for conversational
TOP_P = 0.9

# Retrieval Settings
N_RESULTS_DEFAULT = 3              # Chunks retrieved per query
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Chunking Settings
CHUNK_SIZE = 800                   # Max characters per chunk
CHUNK_OVERLAP = 200                # Overlap between chunks

# Storage Settings
CHROMA_COLLECTION = "documents"
CHROMA_PERSIST_DIR = "./chroma_db"
DOCUMENTS_FOLDER = "./documents"
```

### Switching Machines / Models

When moving to a different machine (e.g., with a larger LLM):

1. Edit `config.py`:
   ```python
   OLLAMA_MODEL = "llama3.1:40b"   # Your larger model
   TEMPERATURE = 0.5                # Can raise for larger models
   N_RESULTS_DEFAULT = 5            # Larger models handle more context
   ```

2. Re-index documents (only needed if changing `EMBEDDING_MODEL` or `CHUNK_SIZE`):
   ```bash
   rm -rf chroma_db
   python preprocess_documents.py
   ```

## Project Structure

```
smart_doc_chat_API/
├── main.py                    # FastAPI application and endpoints
├── config.py                  # Centralized configuration
├── schemas.py                 # Pydantic models for request/response
├── index.html                 # Web chat interface (served at /chat)
├── preprocess_documents.py    # CLI script to index PDFs
├── document_processor.py      # PDF parsing, text cleaning, chunking
├── vector_store.py            # ChromaDB wrapper for embeddings
├── chat_engine.py             # RAG integration with Ollama (standard + streaming)
├── requirements.txt           # Python dependencies
├── documents/                 # PDF storage
└── chroma_db/                 # Vector database storage (auto-created)
```

## How It Works

### RAG Pipeline

**Phase 1: Preprocessing**

1. PDFs placed in `documents/` folder
2. Text extracted page by page using `pypdf`
3. Extracted text cleaned (whitespace normalization, punctuation fixes)
4. Text split into ~800 character chunks with 200 character overlap
5. Each chunk embedded using `BAAI/bge-large-en-v1.5` (1024-dim vectors)
6. Vectors stored in ChromaDB with metadata (filename, page number)

**Phase 2: Querying**

1. User sends question via web UI or API
2. Question embedded using same model
3. ChromaDB finds most similar chunks via cosine similarity
4. Retrieved chunks labeled with page info and sent to LLM as context
5. LLM generates answer based on context only
6. Response streamed token by token with source evidence

## Troubleshooting

### "Cannot connect to Ollama"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull mistral:7b
```

### "No documents indexed"
```bash
python preprocess_documents.py
```

### Slow responses
- Use a smaller model (e.g., `phi4-mini`)
- Reduce `N_RESULTS_DEFAULT` in `config.py`
- Use GPU acceleration with Ollama if available

### Changing embedding model
If you change `EMBEDDING_MODEL` in config, you must re-index:
```bash
rm -rf chroma_db
python preprocess_documents.py
```

## References

- FastAPI: https://fastapi.tiangolo.com
- Ollama: https://ollama.com/library
- ChromaDB: https://docs.trychroma.com
- BGE Embeddings: https://huggingface.co/BAAI/bge-large-en-v1.5
- RAG: https://www.ibm.com/think/topics/retrieval-augmented-generation

## License

Open source. Free for personal and commercial use.
