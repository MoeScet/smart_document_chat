"""
FastAPI application for RAG-based document chat
"""
import logging
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from vector_store import VectorStore
from chat_engine import ChatEngine
from document_processor import process_pdf, get_document_stats
from schemas import (
    ChatRequest,
    ChatResponse,
    DocumentsListResponse,
    DocumentStatsResponse,
    DocumentUploadResponse,
    DocumentDeleteResponse,
    ErrorResponse,
    Source
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
vector_store = None
chat_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global vector_store, chat_engine

    logger.info("Initializing application...")

    try:
        # Initialize vector store
        vector_store = VectorStore()
        logger.info(f"Vector store initialized with {vector_store.get_collection_count()} chunks")

        # Initialize chat engine
        chat_engine = ChatEngine(vector_store)
        logger.info("Chat engine initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Smart Document Chat API",
    description="RAG-based chat API for querying documents using Ollama LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "online",
        "message": "Smart Document Chat API is running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        chunk_count = vector_store.get_collection_count() if vector_store else 0

        # Check Ollama connection
        ollama_status = "online" if chat_engine and chat_engine.check_ollama() else "offline"

        return {
            "status": "healthy",
            "vector_store": "initialized" if vector_store else "not initialized",
            "chunks_indexed": chunk_count,
            "ollama": ollama_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["Chat"]
)
async def chat(request: ChatRequest):
    """
    Chat endpoint - Send a message and get AI response with sources

    Args:
        request: ChatRequest containing message, optional chat history, and retrieval parameters

    Returns:
        ChatResponse with AI response and source citations
    """
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="Chat engine not initialized")

        # Convert chat history to expected format
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.chat_history
        ]

        # Get response from chat engine
        response_text, sources_list = chat_engine.get_response(
            query=request.message,
            chat_history=chat_history,
            n_results=request.n_results
        )

        # Format sources
        formatted_sources = []
        if sources_list:
            for source in sources_list:
                formatted_sources.append(
                    Source(
                        filename=source.get("source", "unknown"),
                        page=source.get("page", 0),
                        chunk_index=source.get("chunk", 0)
                    )
                )

        # Get formatted sources text
        sources_text = None
        if sources_list:
            formatted_source_list = vector_store.format_sources(sources_list)
            sources_text = "Sources:\n" + "\n".join(f"- {src}" for src in formatted_source_list)

        return ChatResponse(
            response=response_text,
            sources=formatted_sources,
            sources_text=sources_text
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get(
    "/documents/indexed",
    response_model=DocumentsListResponse,
    tags=["Documents"]
)
async def get_indexed_documents():
    """
    Get list of all indexed documents

    Returns:
        List of indexed document filenames and total count
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        documents = vector_store.get_indexed_documents()

        return DocumentsListResponse(
            documents=documents,
            total_count=len(documents)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving indexed documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get(
    "/documents/stats",
    response_model=DocumentStatsResponse,
    tags=["Documents"]
)
async def get_document_stats():
    """
    Get statistics about indexed documents

    Returns:
        Statistics including total documents, total chunks, and document list
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        documents = vector_store.get_indexed_documents()
        chunk_count = vector_store.get_collection_count()

        return DocumentStatsResponse(
            total_documents=len(documents),
            total_chunks=chunk_count,
            indexed_documents=documents
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document

    Args:
        file: PDF file to upload

    Returns:
        Upload status with chunks and pages processed
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Check if document already indexed
        if vector_store.is_document_indexed(file.filename):
            raise HTTPException(
                status_code=409,
                detail=f"Document '{file.filename}' is already indexed"
            )

        # Create documents directory if it doesn't exist
        documents_dir = Path("./documents")
        documents_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_path = documents_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing uploaded file: {file.filename}")

        # Process the PDF
        chunks = process_pdf(str(file_path), file.filename)

        if not chunks:
            # Clean up the file if processing failed
            file_path.unlink()
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

        # Get statistics
        stats = get_document_stats(chunks)

        # Add to vector store
        vector_store.add_documents(chunks)

        logger.info(
            f"Successfully indexed '{file.filename}': "
            f"{stats['total_chunks']} chunks from {stats['pages']} pages"
        )

        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            filename=file.filename,
            chunks_added=stats['total_chunks'],
            pages_processed=stats['pages']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@app.delete(
    "/documents/{filename}",
    response_model=DocumentDeleteResponse,
    tags=["Documents"]
)
async def delete_document(filename: str):
    """
    Delete an indexed document

    Args:
        filename: Name of the document to delete

    Returns:
        Deletion status with chunks removed
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        # Check if document is indexed
        if not vector_store.is_document_indexed(filename):
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found in index"
            )

        # Get count before deletion
        initial_count = vector_store.get_collection_count()

        # Delete from vector store
        vector_store.delete_document(filename)

        # Calculate chunks deleted
        final_count = vector_store.get_collection_count()
        chunks_deleted = initial_count - final_count

        # Optionally delete the file from documents folder
        file_path = Path("./documents") / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {filename}")

        logger.info(f"Removed '{filename}' from index ({chunks_deleted} chunks)")

        return DocumentDeleteResponse(
            message="Document deleted successfully",
            filename=filename,
            chunks_deleted=chunks_deleted
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
