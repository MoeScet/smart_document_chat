"""
Document Processor Module
Handles PDF parsing and text chunking for RAG
"""

from pypdf import PdfReader
from typing import List, Dict

def process_pdf(pdf_path: str, filename: str) -> List[Dict]:
    """
    Process a PDF file and return chunks of text with metadata
    
    Args:
        pdf_path: Path to the PDF file
        filename: Name of the file (for metadata)
    
    Returns:
        List of dictionaries containing text chunks and metadata
    """
    chunks = []
    
    try:
        # Read the PDF
        reader = PdfReader(pdf_path)
        
        # Extract text from each page
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            
            if text.strip():  # Only process pages with text
                # Split page into smaller chunks
                page_chunks = split_text(text, chunk_size=800, overlap=200)
                
                # Add metadata to each chunk
                for i, chunk_text in enumerate(page_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'source': filename,
                            'page': page_num,
                            'chunk': i
                        }
                    })
        
        return chunks
    
    except Exception as e:
        raise Exception(f"Error processing PDF {filename}: {str(e)}")


def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Get chunk end position
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence or word
        if end < text_length:
            # Look for sentence break (period, question mark, exclamation)
            for char in ['. ', '? ', '! ', '\n']:
                last_break = text.rfind(char, start, end)
                if last_break != -1:
                    end = last_break + 1
                    break
            else:
                # If no sentence break, try to break at a space
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap if end < text_length else text_length
    
    return chunks


def get_document_stats(chunks: List[Dict]) -> Dict:
    """
    Get statistics about processed documents
    
    Args:
        chunks: List of document chunks
    
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'total_characters': 0,
            'sources': []
        }
    
    sources = list(set(chunk['metadata']['source'] for chunk in chunks))
    total_chars = sum(len(chunk['text']) for chunk in chunks)
    
    return {
        'total_chunks': len(chunks),
        'total_characters': total_chars,
        'sources': sources,
        'avg_chunk_size': total_chars // len(chunks) if chunks else 0
    }