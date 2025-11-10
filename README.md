# Smart Document Chat API

A document chat API using RAG (Retrieval Augmented Generation) to query PDF documents. Built with FastAPI.

## Features

- **Web chat interface** accessible from any browser on your network
- RESTful API for document chat functionality
- Semantic search over indexed documents
- Source citations with each response
- Document management (upload, delete, list)
- Runs locally using Ollama
- Auto-generated API documentation at `/docs`
- Network-accessible from multiple devices

## Tech Stack

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server
- **Ollama**: Local LLM
- **ChromaDB**: Vector database for document embeddings
- **Sentence Transformers**: Embedding generation
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
   ollama pull llama3.1:8b
   ```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `pypdf` - PDF parsing
- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `requests` - HTTP client for Ollama

### 2. Start Ollama

```bash
ollama serve
```

### 3. Run the API Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access points:
- `http://localhost:8000` - API information and status
- `http://localhost:8000/chat` - Chat interface (web UI)
- `http://localhost:8000/docs` - Interactive API documentation
- `http://localhost:8000/redoc` - Alternative API documentation

### 4. Access from Other Devices (Optional)

To access the chat interface from other computers on the same network:

1. **Find your server's IP address:**
   ```bash
   ipconfig
   ```
   Look for IPv4 Address (e.g., `192.168.1.100`)

2. **Configure Windows Firewall:**
   ```bash
   # Run as Administrator
   netsh advfirewall firewall add rule name="Smart Doc Chat" dir=in action=allow protocol=TCP localport=8000
   ```

3. **Access from other devices:**
   - Open browser on another PC/tablet/phone on the same network
   - Navigate to: `http://YOUR_SERVER_IP:8000/chat`
   - Example: `http://192.168.1.100:8000/chat`

## Usage

### Method 1: Preprocessing Script

For batch document indexing:

1. Place PDF files in `documents/` folder

2. Run preprocessing:
   ```bash
   python preprocess_documents.py
   ```

   This extracts text, creates chunks, and stores embeddings in ChromaDB. Already indexed documents are skipped. First run downloads the embedding model (~90MB).

3. Start API server:
   ```bash
   python main.py
   ```

### Method 2: Web Interface

For interactive chat via browser:

1. Start API server:
   ```bash
   python main.py
   ```

2. Open browser and navigate to:
   ```
   http://localhost:8000/chat
   ```

3. Start chatting with your documents through the web interface

### Method 3: API Upload

For programmatic document management:

1. Start API server:
   ```bash
   python main.py
   ```

2. Upload document:
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/document.pdf"
   ```

3. Query documents via API:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is the main topic of this document?",
       "chat_history": [],
       "n_results": 5
     }'
   ```

   Note: POST `/chat` is the API endpoint, while GET `/chat` serves the web interface

## API Endpoints

### Frontend

**GET /chat** - Serve the chat interface
```
http://localhost:8000/chat
```
Opens the web-based chat interface in your browser

### Health & Status

**GET /** - API information and status
```bash
curl http://localhost:8000/
```
Returns API version, status, and available endpoints

**GET /health** - Detailed health status
```bash
curl http://localhost:8000/health
```
Returns vector store status, chunk count, and Ollama connection status

### Chat

**POST /chat** - Send a message and get AI response

Request body:
```json
{
  "message": "What is this document about?",
  "chat_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ],
  "n_results": 5
}
```

Response:
```json
{
  "response": "This document is about...",
  "sources": [
    {
      "filename": "example.pdf",
      "page": 1,
      "chunk_index": 0
    }
  ],
  "sources_text": "Sources:\n- example.pdf (Page 1)"
}
```

Example:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the key points",
    "chat_history": [],
    "n_results": 5
  }'
```

### Document Management

**GET /documents/indexed** - List all indexed documents
```bash
curl http://localhost:8000/documents/indexed
```

**GET /documents/stats** - Get document statistics
```bash
curl http://localhost:8000/documents/stats
```

**POST /documents/upload** - Upload and index a PDF
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

**DELETE /documents/{filename}** - Remove a document
```bash
curl -X DELETE "http://localhost:8000/documents/example.pdf"
```

### Example Queries

- "What is the main topic of this document?"
- "Summarize the key points from page 3"
- "What does the document say about [specific topic]?"
- "Compare the information in section 2 and section 4"

## Project Structure

```
smart_doc_chat/
├── main.py                    # FastAPI application and web server
├── schemas.py                 # Pydantic models for request/response
├── index.html                 # Web chat interface (served at /chat)
├── preprocess_documents.py    # CLI script to index PDFs
├── document_processor.py      # PDF parsing and chunking
├── vector_store.py            # ChromaDB wrapper
├── chat_engine.py             # RAG integration with Ollama
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── documents/                 # PDF storage
└── chroma_db/                 # Vector database storage
```

## Configuration

### Change LLM Model

In `chat_engine.py`, line 15:

```python
def __init__(self, vector_store, model_name: str = "llama3.1:8b"):
```

Available models (install with `ollama pull <model>`):
- `llama3.1:8b` (default)
- `llama3.2`
- `llama3.2:1b`
- `mistral`
- `phi3`

### Adjust Chunk Size

In `document_processor.py`, line 48:

```python
page_chunks = split_text(text, chunk_size=1000, overlap=100)
```

Adjust `chunk_size` (default: 1000) and `overlap` (default: 100) based on document characteristics.

### Change Retrieved Document Count

In `chat_engine.py`, line 46:

```python
def get_response(
    self,
    query: str,
    chat_history: List[Dict] = None,
    n_results: int = 5
)
```

Adjust `n_results` (default: 5) to change the number of document chunks retrieved per query.

## Troubleshooting

### "Cannot connect to Ollama"
Start Ollama server:
```bash
ollama serve
```

### "Model not found"
Download the model:
```bash
ollama pull llama3.1:8b
```

### "No documents indexed"
Run preprocessing:
```bash
python preprocess_documents.py
```

### PDF preprocessing fails
Check the following:
- PDF is not password protected
- pypdf is installed: `pip install pypdf`
- PDF is in the `documents/` folder

### Slow responses
Options to improve performance:
- Use a smaller model: `llama3.2:1b` or `phi3`
- Reduce `n_results` in `chat_engine.py` from 5 to 3
- Use GPU acceleration with Ollama if available

## How It Works

### RAG Pipeline

**Phase 1: Preprocessing**

1. Document Loading
   - PDFs placed in `documents/` folder
   - `preprocess_documents.py` scans for new files

2. Text Extraction
   - `pypdf` extracts text from each page
   - Text prepared for chunking

3. Chunking
   - Text split into ~1000 character chunks with 100 character overlap
   - Each chunk retains metadata (filename, page number)

4. Embedding and Storage
   - Each chunk converted to vector using `all-MiniLM-L6-v2` model
   - Vectors stored in ChromaDB

**Phase 2: Querying**

5. Question Processing
   - User sends POST request to `/chat` endpoint
   - Question converted to vector using same embedding model

6. Semantic Search
   - ChromaDB finds 5 most similar document chunks
   - Uses cosine similarity for matching

7. Response Generation
   - Retrieved chunks, question, and chat history sent to Ollama
   - LLM generates answer based on context
   - Source citations extracted from chunk metadata

### RAG vs Fine-tuning

- No training required
- Immediate integration of new documents
- Verifiable source citations
- Lower resource requirements
- Simple knowledge updates

## References

- FastAPI: https://fastapi.tiangolo.com
- Pydantic: https://docs.pydantic.dev
- Ollama: https://ollama.com/library
- ChromaDB: https://docs.trychroma.com
- RAG: https://www.ibm.com/think/topics/retrieval-augmented-generation

## Extensions

Potential enhancements:

1. Add support for additional file types (.txt, .docx, .csv, .md)
2. Implement multiple collections for document organization
3. Export conversation history (JSON/PDF)
4. Add authentication (JWT/OAuth)
5. Implement rate limiting
6. Add conversation persistence (database storage)
7. Scheduled document re-indexing
8. Streaming responses (SSE)
9. Response caching (Redis)
10. OCR for image-based PDFs
11. Web search integration

## License

Open source. Free for personal and commercial use.