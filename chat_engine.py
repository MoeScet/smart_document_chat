"""
Chat Engine Module
Integrates Ollama LLM with RAG for document-based question answering
"""

from typing import List, Dict, Tuple
import requests
import json

class ChatEngine:
    """
    Handles chat interactions using Ollama and RAG
    """
    
    def __init__(self, vector_store, model_name: str = "llama3.1:8b"):
        """
        Initialize the chat engine
        
        Args:
            vector_store: VectorStore instance for document retrieval
            model_name: Name of the Ollama model to use
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Test Ollama connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print(f"✅ Connected to Ollama")
            else:
                print("⚠️ Warning: Ollama might not be running properly")
        except requests.exceptions.ConnectionError:
            print("❌ Error: Cannot connect to Ollama. Make sure Ollama is running!")
            print("   Run: ollama serve")

    def check_ollama(self) -> bool:
        """
        Check if Ollama is running and accessible

        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False
    
    def get_response(
        self,
        query: str,
        chat_history: List[Dict] = None,
        n_results: int = 5
    ) -> Tuple[str, List[Dict]]:
        """
        Get a response to a user query using RAG

        Args:
            query: User's question
            chat_history: Previous messages in the conversation
            n_results: Number of document chunks to retrieve

        Returns:
            Tuple of (response_text, metadata_list)
            metadata_list contains dicts with 'source', 'page', 'chunk_index' keys
        """
        # Step 1: Retrieve relevant documents
        relevant_docs, metadatas = self.vector_store.search(query, n_results=n_results)

        if not relevant_docs:
            return "I don't have any documents to reference. Please upload some documents first.", []

        # Step 2: Build the context from retrieved documents
        # Join documents with clear separation but no numbering
        context = "\n\n---\n\n".join(relevant_docs)

        # Step 3: Build the prompt
        prompt = self._build_prompt(query, context, chat_history)

        # Step 4: Get response from Ollama
        try:
            response_text = self._call_ollama(prompt)
        except Exception as e:
            return f"Error getting response from Ollama: {str(e)}", []

        # Step 5: Return response with raw metadata
        # The metadata list contains dicts that can be used by different interfaces
        return response_text, metadatas
    
    def _build_prompt(
        self, 
        query: str, 
        context: str, 
        chat_history: List[Dict] = None
    ) -> str:
        """
        Build the prompt for the LLM
        
        Args:
            query: User's question
            context: Retrieved document context
            chat_history: Previous conversation messages
        
        Returns:
            Formatted prompt string
        """
        # System message
        system_msg = """You are a helpful AI assistant that answers questions based on provided documents.

IMPORTANT RULES:
- Answer questions using ONLY the information from the provided documents
- Do NOT mention document numbers (like "Document 1", "Document 2", etc.) in your response
- Answer naturally as if the information is from a single coherent source
- For summarization questions ("what is this about?", "summarize"), provide a comprehensive overview of the content
- If the answer is not in the documents, say "I cannot find that information in the uploaded documents"
- Be concise but thorough
- If asked about something not in the documents, politely decline and explain what information IS available"""
        
        # Build chat history context
        history_text = ""
        if chat_history:
            for msg in chat_history[-3:]:  # Last 3 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"\n{role.upper()}: {content}"
        
        # Complete prompt
        prompt = f"""{system_msg}

PREVIOUS CONVERSATION:
{history_text if history_text else "(No previous conversation)"}

RELEVANT DOCUMENTS:
{context}

USER QUESTION: {query}

ASSISTANT ANSWER:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate a response
        
        Args:
            prompt: The complete prompt to send
        
        Returns:
            Generated response text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response generated")
            
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure it's running with: ollama serve")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. Try a smaller model or simpler question.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama: {str(e)}")
    
    def change_model(self, model_name: str):
        """
        Change the Ollama model being used
        
        Args:
            model_name: Name of the new model
        """
        self.model_name = model_name
        print(f"✅ Switched to model: {model_name}")