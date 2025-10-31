"""
Document Preprocessing Script

This script processes PDF files from the 'documents' folder and indexes them
into the ChromaDB vector store. Run this script whenever you add new PDFs.

Usage:
    python preprocess_documents.py

Features:
    - Automatically finds all PDFs in the documents folder
    - Skips already indexed documents (deduplication)
    - Shows processing progress
    - Provides summary of indexed documents
"""

import os
import sys
from pathlib import Path
from document_processor import process_pdf
from vector_store import VectorStore

# Configuration
# This is where you place your PDF files
DOCUMENTS_FOLDER = "./documents"

# Colors for terminal output (cross-platform)
class Colors:
    """ANSI color codes for pretty terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print script header with instructions"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  Document Preprocessing Script{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    print(f"Scanning folder: {Colors.BLUE}{DOCUMENTS_FOLDER}{Colors.END}")
    print()

def get_pdf_files(folder_path):
    """
    Get all PDF files from the documents folder

    Args:
        folder_path: Path to the documents folder

    Returns:
        List of PDF file paths
    """
    # Convert to Path object for easier file handling
    folder = Path(folder_path)

    # Check if folder exists
    if not folder.exists():
        print(f"{Colors.RED}[ERROR] Folder '{folder_path}' does not exist!{Colors.END}")
        print(f"{Colors.YELLOW}Creating folder: {folder_path}{Colors.END}")
        folder.mkdir(parents=True, exist_ok=True)
        return []

    # Find all PDF files (case-insensitive)
    # This searches for both .pdf and .PDF extensions
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))

    return pdf_files

def process_documents():
    """
    Main function to process all documents

    This function:
    1. Initializes the vector store
    2. Scans the documents folder for PDFs
    3. Checks which PDFs are already indexed
    4. Processes new PDFs and adds them to the vector store
    5. Provides a summary report
    """

    print_header()

    # Step 1: Initialize vector store
    # This connects to the persistent ChromaDB storage
    print(f"{Colors.BOLD}Step 1: Initializing vector store...{Colors.END}")
    try:
        vector_store = VectorStore()
        print(f"{Colors.GREEN}[OK] Vector store initialized{Colors.END}\n")
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Error initializing vector store: {str(e)}{Colors.END}")
        sys.exit(1)

    # Step 2: Get already indexed documents
    # This prevents re-processing documents that are already in the database
    indexed_docs = vector_store.get_indexed_documents()
    if indexed_docs:
        print(f"{Colors.BOLD}Already indexed documents:{Colors.END}")
        for doc in indexed_docs:
            print(f"  - {doc}")
        print()
    else:
        print(f"{Colors.YELLOW}[INFO] No documents currently indexed{Colors.END}\n")

    # Step 3: Scan documents folder
    print(f"{Colors.BOLD}Step 2: Scanning for PDF files...{Colors.END}")
    pdf_files = get_pdf_files(DOCUMENTS_FOLDER)

    if not pdf_files:
        print(f"{Colors.YELLOW}[WARNING] No PDF files found in '{DOCUMENTS_FOLDER}'{Colors.END}")
        print(f"{Colors.YELLOW}[INFO] Place PDF files in the '{DOCUMENTS_FOLDER}' folder and run this script again{Colors.END}")
        return

    print(f"{Colors.GREEN}Found {len(pdf_files)} PDF file(s){Colors.END}\n")

    # Step 4: Process each PDF
    print(f"{Colors.BOLD}Step 3: Processing documents...{Colors.END}\n")

    # Track statistics
    new_documents = []
    skipped_documents = []
    failed_documents = []
    total_chunks = 0

    for pdf_path in pdf_files:
        # Get just the filename (not the full path)
        filename = pdf_path.name

        # Check if this document is already indexed
        # This avoids duplicate processing
        if vector_store.is_document_indexed(filename):
            print(f"{Colors.YELLOW}[SKIP] Skipping (already indexed): {filename}{Colors.END}")
            skipped_documents.append(filename)
            continue

        # Process this new document
        print(f"{Colors.BLUE}Processing: {filename}{Colors.END}")

        try:
            # Extract text from PDF and split into chunks
            # process_pdf returns a list of text chunks with metadata
            chunks = process_pdf(str(pdf_path), filename)

            # Add chunks to vector store
            # This creates embeddings and stores them in ChromaDB
            vector_store.add_documents(chunks)

            # Record success
            new_documents.append(filename)
            total_chunks += len(chunks)

            print(f"{Colors.GREEN}   [OK] Indexed {len(chunks)} chunks{Colors.END}")

        except Exception as e:
            # Handle errors gracefully
            print(f"{Colors.RED}   [ERROR] Error processing {filename}: {str(e)}{Colors.END}")
            failed_documents.append((filename, str(e)))

    # Step 5: Print summary report
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  Processing Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

    if new_documents:
        print(f"{Colors.GREEN}[OK] Successfully indexed: {len(new_documents)} document(s){Colors.END}")
        for doc in new_documents:
            print(f"   - {doc}")
        print(f"\n   Total chunks created: {total_chunks}")
        print()

    if skipped_documents:
        print(f"{Colors.YELLOW}[SKIP] Skipped (already indexed): {len(skipped_documents)} document(s){Colors.END}")
        for doc in skipped_documents:
            print(f"   - {doc}")
        print()

    if failed_documents:
        print(f"{Colors.RED}[FAILED] Failed: {len(failed_documents)} document(s){Colors.END}")
        for doc, error in failed_documents:
            print(f"   - {doc}: {error}")
        print()

    # Show final state of vector store
    all_indexed = vector_store.get_indexed_documents()
    total_count = vector_store.get_collection_count()

    print(f"{Colors.BOLD}Vector Store Status:{Colors.END}")
    print(f"   Total documents: {len(all_indexed)}")
    print(f"   Total chunks: {total_count}")
    print()

    print(f"{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}[OK] Preprocessing complete!{Colors.END}")
    print(f"{Colors.GREEN}{'='*60}{Colors.END}\n")

    # Provide next steps
    if new_documents:
        print(f"{Colors.BOLD}Next steps:{Colors.END}")
        print(f"   1. Run the Streamlit app: streamlit run app.py")
        print(f"   2. Start asking questions about your documents!")
        print()

if __name__ == "__main__":
    """
    Entry point when script is run directly

    This allows the script to be imported as a module without auto-executing,
    but will run process_documents() when executed directly.
    """
    try:
        process_documents()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print(f"\n\n{Colors.YELLOW}[WARNING] Processing interrupted by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        # Catch any unexpected errors
        print(f"\n{Colors.RED}[ERROR] Unexpected error: {str(e)}{Colors.END}")
        sys.exit(1)
