"""
Chat Engine Module
Integrates Ollama LLM with RAG for document-based question answering
"""

from typing import List, Dict, Tuple
import httpx
import json
from config import OLLAMA_MODEL, OLLAMA_URL, TEMPERATURE, TOP_P, N_RESULTS_DEFAULT

class ChatEngine:
    """
    Handles chat interactions using Ollama and RAG
    """

    def __init__(self, vector_store, model_name: str = None):
        """
        Initialize the chat engine

        Args:
            vector_store: VectorStore instance for document retrieval
            model_name: Name of the Ollama model to use
        """
        self.vector_store = vector_store
        self.model_name = model_name or OLLAMA_MODEL
        self.ollama_url = f"{OLLAMA_URL}/api/generate"
        
        # Test Ollama connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                print("Connected to Ollama")
            else:
                print("Warning: Ollama might not be running properly")
        except httpx.ConnectError:
            print("Error: Cannot connect to Ollama. Make sure Ollama is running!")
            print("   Run: ollama serve")

    def check_ollama(self) -> bool:
        """
        Check if Ollama is running and accessible

        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{OLLAMA_URL}/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
    
    async def get_response(
        self,
        query: str,
        chat_history: List[Dict] = None,
        n_results: int = None
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
        n_results = n_results or N_RESULTS_DEFAULT

        # Step 1: Retrieve relevant documents
        relevant_docs, metadatas, distances = self.vector_store.search(query, n_results=n_results)

        if not relevant_docs:
            return "I don't have any documents to reference. Please upload some documents first.", []

        # Step 2: Build context with page labels
        context_parts = []
        for doc, meta in zip(relevant_docs, metadatas):
            page = meta.get("page", "?")
            source = meta.get("source", "Unknown")
            context_parts.append(f"[{source} - Page {page}]:\n{doc}")
        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Build the prompt
        prompt = self._build_prompt(query, context, chat_history)

        # Step 4: Get response from Ollama
        try:
            response_text = await self._call_ollama(prompt)
        except Exception as e:
            return f"Error getting response from Ollama: {str(e)}", []

        # Step 5: Build source list with chunk text included
        sources = []
        for doc, meta in zip(relevant_docs, metadatas):
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0),
                "chunk": meta.get("chunk", 0),
                "text": doc
            })

        return response_text, sources
    
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
        system_msg = """You are a helpful assistant. Answer the user's question using ONLY the provided documents below. Read ALL the documents carefully before answering.

Rules:
- Use ONLY information found in the documents. Do not add outside knowledge.
- If the documents contain the answer, provide it. Read every section thoroughly.
- If the answer is not in the documents, say so.
- Do NOT reference page numbers, document labels, or section headers in your answer.
- Be concise but thorough. Answer every part of the question."""

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

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    async def get_response_stream(
        self,
        query: str,
        chat_history: List[Dict] = None,
        n_results: int = None
    ):
        """
        Stream a response token by token, then yield sources at the end.

        Yields JSON strings:
            {"type": "token", "content": "word"}
            {"type": "sources", "sources": [...], "sources_text": "..."}
            {"type": "done"}
        """
        n_results = n_results or N_RESULTS_DEFAULT

        # Step 1: Retrieve relevant documents
        relevant_docs, metadatas, distances = self.vector_store.search(query, n_results=n_results)

        if not relevant_docs:
            yield json.dumps({"type": "token", "content": "I don't have any documents to reference. Please upload some documents first."})
            yield json.dumps({"type": "done"})
            return

        # Step 2: Build context with page labels
        context_parts = []
        for doc, meta in zip(relevant_docs, metadatas):
            page = meta.get("page", "?")
            source = meta.get("source", "Unknown")
            context_parts.append(f"[{source} - Page {page}]:\n{doc}")
        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Build the prompt
        prompt = self._build_prompt(query, context, chat_history)

        # Step 4: Stream from Ollama
        try:
            async for token in self._call_ollama_stream(prompt):
                yield json.dumps({"type": "token", "content": token})
        except Exception as e:
            yield json.dumps({"type": "token", "content": f"Error: {str(e)}"})
            yield json.dumps({"type": "done"})
            return

        # Step 5: Build and yield sources
        sources = []
        for doc, meta in zip(relevant_docs, metadatas):
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0),
                "text": doc
            })

        # Build sources_text
        parts = []
        for src in sources:
            snippet = src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"]
            parts.append(f"[{src['source']} - Page {src['page']}]\n\"{snippet}\"")
        sources_text = "Sources:\n" + "\n\n".join(parts)

        yield json.dumps({
            "type": "sources",
            "sources": [
                {"filename": s["source"], "page": s["page"], "text": s["text"]}
                for s in sources
            ],
            "sources_text": sources_text
        })
        yield json.dumps({"type": "done"})

    async def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate a complete response (non-blocking).

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
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(self.ollama_url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "No response generated")

        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama. Make sure it's running with: ollama serve")
        except httpx.ReadTimeout:
            raise Exception("Ollama request timed out. Try a smaller model or simpler question.")
        except httpx.HTTPError as e:
            raise Exception(f"Error calling Ollama: {str(e)}")

    async def _call_ollama_stream(self, prompt: str):
        """
        Call Ollama API with streaming enabled (non-blocking). Yields tokens as they arrive.

        Args:
            prompt: The complete prompt to send

        Yields:
            Individual token strings
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", self.ollama_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break

        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama. Make sure it's running with: ollama serve")
        except httpx.ReadTimeout:
            raise Exception("Ollama request timed out. Try a smaller model or simpler question.")
        except httpx.HTTPError as e:
            raise Exception(f"Error calling Ollama: {str(e)}")
    
    def change_model(self, model_name: str):
        """
        Change the Ollama model being used
        
        Args:
            model_name: Name of the new model
        """
        self.model_name = model_name
        print(f"Switched to model: {model_name}")