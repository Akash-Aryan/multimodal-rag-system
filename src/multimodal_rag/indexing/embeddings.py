"""Embedding generation and management."""

from typing import Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..config import RAGConfig


class EmbeddingManager:
    """Manage embeddings for multimodal content."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_model = None
        self.image_model = None
    
    def _load_text_model(self):
        if self.text_model is None:
            self.text_model = SentenceTransformer(self.config.embedding.text_model)
    
    async def generate_embeddings(self, content_data: Dict[str, Any]) -> np.ndarray:
        """Generate embeddings for content chunks."""
        self._load_text_model()
        
        chunks = content_data.get('chunks', [])
        if not chunks:
            return np.array([])
        
        texts = [chunk.get('text', '') for chunk in chunks]
        embeddings = self.text_model.encode(texts)
        
        logger.info(f"Generated embeddings for {len(texts)} chunks")
        return embeddings
