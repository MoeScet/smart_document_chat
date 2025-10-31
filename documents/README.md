# Documents Folder

This folder contains PDF files that will be indexed by the RAG system.

## How to Add Documents

1. **Place PDF files in this folder**
   ```
   documents/
   ├── washing_machine_manual.pdf
   ├── troubleshooting_guide.pdf
   └── parts_catalog.pdf
   ```

2. **Run the preprocessing script**
   ```bash
   python preprocess_documents.py
   ```

3. **Start or refresh the Streamlit app**
   ```bash
   streamlit run app.py
   ```

OR
   
   ```bash
   python -m streamlit run app.py
   ```
## Supported File Types

- `.pdf` - PDF documents
- `.PDF` - PDF documents (case-insensitive)

## File Naming Best Practices

- Use descriptive names: `washing_machine_manual.pdf` (not `doc1.pdf`)
- Avoid special characters: Use `_` instead of spaces
- Keep names concise but clear

## What Happens During Preprocessing

The `preprocess_documents.py` script will:

1. Scan this folder for PDF files
2. Check which files are already indexed (skips duplicates)
3. Extract text from new PDFs
4. Split text into chunks (~1000 characters each)
5. Generate embeddings (384-dimensional vectors)
6. Store in ChromaDB (`chroma_db/` folder)
7. Show processing summary

## Deduplication

The system automatically:
- **Skips** files that are already indexed
- **Processes** only new files
- **Prevents** duplicate indexing

If you update a PDF file:
1. Delete it from the vector store first (see below)
2. Then re-run preprocessing

## How to Remove Documents

### Option 1: Using Python
```python
from vector_store import VectorStore

vector_store = VectorStore()

# List indexed documents
docs = vector_store.get_indexed_documents()
print(docs)

# Delete specific document
vector_store.delete_document("washing_machine_manual.pdf")

# Or clear all documents
vector_store.clear()
```

### Option 2: Delete ChromaDB folder
```bash
# This removes ALL indexed documents
rm -rf chroma_db/
```

## Storage Information

**Documents folder:**
- Contains original PDF files
- Not indexed by git (see `.gitignore`)
- Can be large (MB to GB)

**ChromaDB folder:**
- Contains processed embeddings
- Typically 2-5 MB per document
- Generated from PDFs (can be regenerated)

## Troubleshooting

### "No PDF files found"
- Check that PDFs are directly in `documents/` folder (not subfolders)
- Verify file extensions are `.pdf` or `.PDF`

### "Error processing PDF"
- PDF might be corrupted
- PDF might be password-protected
- PDF might be image-based (no text layer)

### "Already indexed" warning
- This is normal - the file was previously processed
- To re-process, delete from vector store first

## Example Workflow

```bash
# Initial setup
cd "smart_doc_chat"

# Add PDFs to documents folder
cp ~/Downloads/*.pdf documents/

# Index the documents
python preprocess_documents.py

# Output:
# [OK] Successfully indexed: 3 document(s)
#    - washing_machine_manual.pdf
#    - troubleshooting_guide.pdf
#    - parts_catalog.pdf
#    Total chunks created: 247

# Start the app
streamlit run app.py
or
python -m streamlit run app.py

# Now users can ask questions!
```

## Questions?

See the main README.md for more information about the RAG system.
