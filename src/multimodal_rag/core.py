"""Core multimodal RAG system implementation."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
from loguru import logger

from .config import RAGConfig
from .ingestion.document_processor import DocumentProcessor
from .ingestion.image_processor import ImageProcessor
from .ingestion.audio_processor import AudioProcessor
from .indexing.vector_store import VectorStore
from .indexing.embeddings import EmbeddingManager
from .retrieval.retriever import MultimodalRetriever
from .models.llm_manager import LLMManager


class MultimodalRAG:
    """
    Main class for the multimodal RAG system.
    
    Handles ingestion, indexing, and retrieval of multimodal content
    including text documents, images, and audio files.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the multimodal RAG system."""
        self.config = config or RAGConfig()
        self.config.ensure_directories()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.audio_processor = AudioProcessor(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.vector_store = VectorStore(self.config)
        self.retriever = MultimodalRetriever(
            self.vector_store, 
            self.embedding_manager,
            self.config
        )
        self.llm_manager = LLMManager(self.config)
        
        logger.info("Multimodal RAG system initialized")
    
    async def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest a single file into the system.
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            Dict containing ingestion results and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower().lstrip('.')
        
        logger.info(f"Ingesting file: {file_path}")
        
        # Process based on file type
        if file_extension in ['pdf', 'docx', 'txt', 'md']:
            content_data = await self.document_processor.process_file(file_path)
        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            content_data = await self.image_processor.process_file(file_path)
        elif file_extension in ['wav', 'mp3', 'mp4', 'flac', 'm4a']:
            content_data = await self.audio_processor.process_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Generate embeddings and store
        embeddings = await self.embedding_manager.generate_embeddings(content_data)
        storage_result = await self.vector_store.add_documents(content_data, embeddings)
        
        result = {
            'file_path': str(file_path),
            'file_type': file_extension,
            'content_chunks': len(content_data.get('chunks', [])),
            'storage_ids': storage_result.get('ids', []),
            'status': 'success'
        }
        
        logger.info(f"Successfully ingested {file_path}")
        return result
    
    async def ingest_directory(self, directory_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Ingest all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing files to ingest
            
        Returns:
            List of ingestion results for each file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        supported_extensions = set(self.config.document_processing.supported_formats)
        files_to_process = []
        
        # Find all supported files
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower().lstrip('.') in supported_extensions:
                files_to_process.append(file_path)
        
        logger.info(f"Found {len(files_to_process)} files to process in {directory_path}")
        
        # Process files concurrently
        results = []
        tasks = [self.ingest_file(file_path) for file_path in files_to_process]
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                results.append({
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    async def query(self, 
                   query: str, 
                   query_type: str = "text",
                   top_k: Optional[int] = None,
                   include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query: Query text or path to query file (for image/audio queries)
            query_type: Type of query ("text", "image", "audio")
            top_k: Number of results to retrieve (overrides config)
            include_sources: Whether to include source documents in response
            
        Returns:
            Dict containing the generated response and metadata
        """
        top_k = top_k or self.config.top_k
        
        logger.info(f"Processing {query_type} query: {query[:100]}...")
        
        # Retrieve relevant documents
        retrieval_results = await self.retriever.retrieve(
            query=query,
            query_type=query_type,
            top_k=top_k
        )
        
        # Generate response using LLM
        response = await self.llm_manager.generate_response(
            query=query,
            context_documents=retrieval_results.get('documents', []),
            query_type=query_type
        )
        
        result = {
            'query': query,
            'query_type': query_type,
            'response': response.get('response', ''),
            'retrieved_count': len(retrieval_results.get('documents', [])),
            'confidence_scores': retrieval_results.get('scores', []),
        }
        
        if include_sources:
            result['sources'] = retrieval_results.get('documents', [])
        
        logger.info("Query processed successfully")
        return result
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and status."""
        vector_stats = await self.vector_store.get_stats()
        
        return {
            'total_documents': vector_stats.get('total_documents', 0),
            'vector_dimension': self.config.vector_store.dimension,
            'supported_formats': self.config.document_processing.supported_formats,
            'llm_model': self.config.llm.model_name,
            'embedding_models': {
                'text': self.config.embedding.text_model,
                'image': self.config.embedding.image_model,
            }
        }
    
    async def clear_index(self) -> None:
        """Clear the vector index and all stored documents."""
        await self.vector_store.clear()
        logger.info("Vector index cleared")
    
    async def save_index(self, path: Optional[str] = None) -> None:
        """Save the vector index to disk."""
        await self.vector_store.save(path)
        logger.info(f"Index saved to {path or 'default location'}")
    
    async def load_index(self, path: Optional[str] = None) -> None:
        """Load the vector index from disk."""
        await self.vector_store.load(path)
        logger.info(f"Index loaded from {path or 'default location'}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup resources if needed
        pass
