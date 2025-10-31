"""
Pydantic models for API request/response validation
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message/question", min_length=1)
    chat_history: List[Message] = Field(
        default=[],
        description="Previous conversation history"
    )
    n_results: int = Field(
        default=5,
        description="Number of relevant documents to retrieve",
        ge=1,
        le=20
    )


class Source(BaseModel):
    """Document source information"""
    filename: str
    page: int
    chunk_index: int


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI assistant's response")
    sources: List[Source] = Field(
        default=[],
        description="List of document sources used"
    )
    sources_text: Optional[str] = Field(
        None,
        description="Formatted sources as text"
    )


class DocumentInfo(BaseModel):
    """Information about an indexed document"""
    filename: str
    chunk_count: Optional[int] = None


class DocumentsListResponse(BaseModel):
    """Response model for documents list endpoint"""
    documents: List[str]
    total_count: int


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics endpoint"""
    total_documents: int
    total_chunks: int
    indexed_documents: List[str]


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoint"""
    message: str
    filename: str
    chunks_added: int
    pages_processed: int


class DocumentDeleteResponse(BaseModel):
    """Response model for document delete endpoint"""
    message: str
    filename: str
    chunks_deleted: int
