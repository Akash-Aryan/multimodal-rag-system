"""LLM integration for response generation."""

from typing import Dict, List, Any
import requests
import json
from loguru import logger

from ..config import RAGConfig


class LLMManager:
    """Manage LLM interactions for response generation."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    async def generate_response(self, query: str, context_documents: List[Dict], query_type: str = "text") -> Dict[str, Any]:
        """Generate response using LLM with retrieved context."""
        
        # Build context from retrieved documents
        context = self._build_context(context_documents)
        
        # Create prompt
        prompt = self._create_prompt(query, context, query_type)
        
        # Generate response based on provider
        if self.config.llm.provider == "ollama":
            response = await self._generate_with_ollama(prompt)
        else:
            response = "LLM response generation not implemented for this provider."
        
        return {"response": response, "context_used": len(context_documents)}
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for doc in documents:
            if "document" in doc and "text" in doc["document"]:
                context_parts.append(doc["document"]["text"])
        
        return "\\n\\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, query_type: str) -> str:
        """Create prompt for LLM."""
        return f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
    
    async def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.llm.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.llm.temperature,
                        "num_predict": self.config.llm.max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "Error generating response from LLM."
                
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return "Error connecting to LLM service."
