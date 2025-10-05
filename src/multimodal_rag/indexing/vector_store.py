"""Vector store implementation using FAISS."""

from typing import Dict, List, Any, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path
from loguru import logger

from ..config import RAGConfig


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, config: RAGConfig):
        """Initialize vector store."""
        self.config = config
        self.index = None
        self.document_store = {}
        self.metadata_store = {}
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        dimension = self.config.vector_store.dimension
        
        if self.config.vector_store.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.config.vector_store.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        logger.info(f"Initialized FAISS index: {self.config.vector_store.index_type}")
    
    async def add_documents(self, content_data: Dict[str, Any], embeddings: np.ndarray) -> Dict[str, Any]:
        """Add documents and embeddings to the store."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Store document content and metadata
        doc_ids = []
        for i, chunk in enumerate(content_data.get('chunks', [])):
            doc_id = start_id + i
            self.document_store[doc_id] = chunk
            self.metadata_store[doc_id] = content_data
            doc_ids.append(doc_id)
        
        logger.info(f"Added {len(doc_ids)} documents to vector store")
        
        return {"ids": doc_ids, "status": "success"}
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar documents."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.document_store:
                results.append({
                    "document": self.document_store[idx],
                    "score": float(score),
                    "metadata": self.metadata_store[idx]
                })
        
        return {"documents": results, "scores": scores[0].tolist()}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_documents": self.index.ntotal if self.index else 0,
            "dimension": self.config.vector_store.dimension,
            "index_type": self.config.vector_store.index_type
        }
    
    async def save(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        save_path = Path(path or self.config.vector_store.persist_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save document and metadata stores
        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump(self.document_store, f)
        
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)
        
        logger.info(f"Saved vector store to {save_path}")
    
    async def load(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        load_path = Path(path or self.config.vector_store.persist_directory)
        
        if not load_path.exists():
            logger.warning(f"No saved index found at {load_path}")
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load document and metadata stores
        with open(load_path / "documents.pkl", "rb") as f:
            self.document_store = pickle.load(f)
        
        with open(load_path / "metadata.pkl", "rb") as f:
            self.metadata_store = pickle.load(f)
        
        logger.info(f"Loaded vector store from {load_path}")
    
    async def clear(self) -> None:
        """Clear all stored documents and embeddings."""
        self._initialize_index()
        self.document_store.clear()
        self.metadata_store.clear()
        logger.info("Cleared vector store")
