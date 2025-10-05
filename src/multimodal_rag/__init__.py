"""
Multimodal RAG System - A comprehensive retrieval-augmented generation system
for processing and querying diverse data formats including documents, images, and audio.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import MultimodalRAG
from .config import RAGConfig

__all__ = ["MultimodalRAG", "RAGConfig"]
