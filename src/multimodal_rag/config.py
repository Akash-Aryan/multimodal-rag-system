"""Configuration management for the multimodal RAG system."""

from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    provider: str = Field(default="faiss", description="Vector store provider (faiss, chroma, pinecone)")
    dimension: int = Field(default=384, description="Vector dimension")
    index_type: str = Field(default="IndexFlatIP", description="FAISS index type")
    persist_directory: str = Field(default="./data/embeddings", description="Directory to persist embeddings")


class LLMConfig(BaseModel):
    """Configuration for LLM."""
    provider: str = Field(default="ollama", description="LLM provider (ollama, huggingface)")
    model_name: str = Field(default="llama2", description="Model name")
    model_path: Optional[str] = Field(default=None, description="Local model path")
    api_key: Optional[str] = Field(default=None, description="API key if required")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    text_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Text embedding model")
    image_model: str = Field(default="clip-ViT-B-32", description="Image embedding model")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "txt", "md", "jpg", "png", "jpeg", "wav", "mp3"],
        description="Supported file formats"
    )


class RAGConfig(BaseModel):
    """Main configuration class for the RAG system."""
    
    # Data directories
    data_dir: str = Field(default="./data", description="Root data directory")
    raw_data_dir: str = Field(default="./data/raw", description="Raw data directory")
    processed_data_dir: str = Field(default="./data/processed", description="Processed data directory")
    
    # Model configurations
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    
    # Retrieval settings
    top_k: int = Field(default=5, description="Number of top results to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    
    # Audio processing
    whisper_model: str = Field(default="base", description="Whisper model size")
    
    # Image processing
    ocr_enabled: bool = Field(default=True, description="Enable OCR for images")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RAGConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.vector_store.persist_directory,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
