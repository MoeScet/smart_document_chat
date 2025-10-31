"""
Smart Document Chat - Main Application
A simple RAG (Retrieval Augmented Generation) chatbot that lets you chat with your PDFs
"""

import streamlit as st
from vector_store import VectorStore
from chat_engine import ChatEngine

# Page configuration
st.set_page_config(
    page_title="Smart Document Chat",
    page_icon=":books:",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize vector store on app startup (not just when uploading)
# This allows us to load existing documents from persistent storage
if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = VectorStore()
        # Check if there are already indexed documents
        indexed_docs = st.session_state.vector_store.get_indexed_documents()
        if indexed_docs:
            st.session_state.documents_loaded = True
    except Exception as e:
        st.session_state.vector_store = None
        st.error(f"Error initializing vector store: {str(e)}")

if "chat_engine" not in st.session_state:
    # Initialize chat engine if we have a vector store
    if st.session_state.vector_store is not None:
        st.session_state.chat_engine = ChatEngine(st.session_state.vector_store)
    else:
        st.session_state.chat_engine = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# App title and description
st.title("Smart Document Chat")
st.markdown("Ask questions about your indexed documents using AI!")

# Sidebar for document information and controls
with st.sidebar:
    st.header("Knowledge Base")

    # Show indexed documents section (read-only)
    if st.session_state.vector_store:
        # Get list of all indexed documents from the persistent store
        indexed_docs = st.session_state.vector_store.get_indexed_documents()

        if indexed_docs:
            # Show summary statistics
            total_chunks = st.session_state.vector_store.get_collection_count()
            st.success(f"{len(indexed_docs)} document(s) indexed")
            st.info(f"Total chunks: {total_chunks}")

            # Display each indexed document (read-only, no delete)
            st.divider()
            st.subheader("Indexed Documents")
            for doc in indexed_docs:
                st.text(f"- {doc}")

            # Instructions for adding/removing documents
            st.divider()
            st.markdown("""
            **To manage documents:**

            **Add new documents:**
            1. Place PDF files in `documents/` folder
            2. Run: `python preprocess_documents.py`

            **Remove documents:**
            - Use the preprocessing script with delete option
            - Or manually clear the database
            """)

        else:
            # No documents indexed yet
            st.warning("No documents indexed")
            st.markdown("""
            **To get started:**

            1. Place PDF files in the `documents/` folder
            2. Run the preprocessing script:
               ```
               python preprocess_documents.py
               ```
            3. Refresh this page
            """)

    # Clear chat history button
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not st.session_state.documents_loaded:
    st.info("No documents indexed yet. Please run the preprocessing script to index documents (see sidebar for instructions).")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.chat_engine.get_response(
                        prompt,
                        st.session_state.messages[:-1]  # Pass chat history without current message
                    )
                    st.markdown(response)

                    # Show sources
                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit, LangChain, ChromaDB, and Ollama 🚀
    </div>
    """,
    unsafe_allow_html=True
)