"""
Document Processor Module
Handles PDF parsing and text chunking for RAG
"""

from pypdf import PdfReader
from typing import List, Dict
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

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
            raw_text = page.extract_text()
            text = clean_extracted_text(raw_text)

            if text.strip():  # Only process pages with text
                # Split page into smaller chunks
                page_chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                
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


def clean_extracted_text(text: str) -> str:
    """
    Clean noisy text from PDF extraction.

    pypdf often extracts text with each word on its own line separated
    by spaces/newlines (e.g. "Thet\\n \\nZin" instead of "Thet Zin").
    This function collapses that noise into clean readable text.

    Args:
        text: Raw extracted text from pypdf

    Returns:
        Cleaned text with proper spacing
    """
    # Replace the common pypdf pattern: \n \n between words -> single space
    text = re.sub(r'\n \n', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up lines that are just whitespace
    text = re.sub(r'\n +\n', '\n\n', text)

    # Fix spacing before punctuation (e.g. "word ,word" -> "word, word")
    text = re.sub(r'\s+([,.])', r'\1', text)

    return text.strip()


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
            'pages': 0,
            'sources': []
        }

    sources = list(set(chunk['metadata']['source'] for chunk in chunks))
    total_chars = sum(len(chunk['text']) for chunk in chunks)
    pages = len(set(chunk['metadata']['page'] for chunk in chunks))

    return {
        'total_chunks': len(chunks),
        'total_characters': total_chars,
        'pages': pages,
        'sources': sources,
        'avg_chunk_size': total_chars // len(chunks) if chunks else 0
    }