# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **Multimodal RAG (Retrieval-Augmented Generation) System** designed for **offline operation** that can ingest, index, and query diverse data formats including documents (PDF, DOCX, TXT, MD), images (with OCR), and audio files (with speech-to-text). The system uses local LLMs via Ollama for complete privacy and offline functionality.

## Essential Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows PowerShell
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
copy .env.example .env
# Edit .env with your configuration
```

### System Dependencies
```bash
# Install Ollama (required for LLM inference)
# Download from https://ollama.ai/
ollama pull llama2  # Pull default model
ollama list         # Verify installation

# Install Tesseract OCR for image processing
# Windows: Download from GitHub releases
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

### Core Operations

#### Data Ingestion
```bash
# Ingest single file
python scripts/ingest.py --file "path/to/document.pdf"

# Ingest entire directory
python scripts/ingest.py --directory "data/raw" --save-index

# Ingest with custom config
python scripts/ingest.py --directory "data/raw" --config "config/custom.yaml"
```

#### Querying
```bash
# Simple text query
python scripts/query.py --query "What are the main topics in the documents?"

# Interactive query mode
python scripts/query.py --interactive

# Query with specific parameters
python scripts/query.py --query "Summarize the findings" --top-k 10 --query-type text
```

#### Development and Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_ingestion.py
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=src tests/

# Code formatting and linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Docker Operations
```bash
# Build and run with Docker Compose
docker-compose build
docker-compose up -d

# View logs
docker-compose logs multimodal-rag

# API endpoint testing
curl http://localhost:8000/query -X POST -H "Content-Type: application/json" -d '{"query": "What is in the documents?", "top_k": 5}'
```

## Code Architecture

### Core System Flow
The system follows a modular pipeline architecture:

1. **Ingestion Layer** (`src/multimodal_rag/ingestion/`):
   - `DocumentProcessor`: Handles PDF, DOCX, TXT, MD files with intelligent text extraction
   - `ImageProcessor`: OCR text extraction from images using Tesseract
   - `AudioProcessor`: Speech-to-text transcription using OpenAI Whisper

2. **Indexing Layer** (`src/multimodal_rag/indexing/`):
   - `EmbeddingManager`: Generates embeddings using sentence-transformers for text and CLIP for images
   - `VectorStore`: FAISS-based vector storage with cosine similarity search

3. **Retrieval Layer** (`src/multimodal_rag/retrieval/`):
   - `MultimodalRetriever`: Unified retrieval interface across all content types

4. **LLM Layer** (`src/multimodal_rag/models/`):
   - `LLMManager`: Ollama integration for local LLM inference

5. **Core Orchestration** (`src/multimodal_rag/core.py`):
   - `MultimodalRAG`: Main class coordinating all components

### Key Design Patterns

#### Async/Await Architecture
All processing operations are asynchronous to handle multiple files efficiently:
- File processing uses ThreadPoolExecutor for CPU-bound tasks
- Concurrent ingestion with `asyncio.as_completed()`
- Non-blocking I/O operations throughout

#### Configuration Management
Centralized configuration via Pydantic models (`config.py`):
- YAML-based configuration with environment variable support
- Type validation and default values
- Hierarchical config structure (VectorStoreConfig, LLMConfig, etc.)

#### Modular Processing Pipeline
Each content type follows the same processing pattern:
1. File validation and type detection
2. Content extraction (text/OCR/transcription)
3. Chunking with overlap for optimal retrieval
4. Metadata extraction and storage
5. Embedding generation and vector storage

### Data Flow Architecture

```
Input Files → Content Processors → Text Chunks → Embedding Generation → Vector Store
                                                        ↓
Query Input → Query Embedding → Similarity Search → Context Retrieval → LLM Response
```

### Critical Implementation Details

#### Vector Storage Strategy
- Uses FAISS IndexFlatIP for inner product similarity (cosine similarity after L2 normalization)
- Documents and metadata stored separately in pickle files for persistence
- Automatic embedding normalization for consistent similarity scoring

#### Text Processing Pipeline
- RecursiveCharacterTextSplitter with configurable chunk size (default 1000) and overlap (default 200)
- Multiple fallback strategies for PDF extraction (pdfplumber → PyPDF2)
- Unicode encoding detection for plain text files

#### Retrieval-Augmented Generation
- Context-aware response generation using retrieved document chunks
- Configurable similarity thresholds and result limits
- Source attribution maintained throughout the pipeline

## Development Guidelines

### Testing Strategy
- Unit tests for each processor type (`tests/test_ingestion.py`)
- Integration tests for end-to-end workflows (`tests/test_retrieval.py`)
- Async test support with `pytest-asyncio`

### Adding New Content Types
1. Create new processor in `src/multimodal_rag/ingestion/`
2. Update `supported_formats` in `config.yaml`
3. Add processing logic to `core.py` file type detection
4. Update embedding manager if needed for new content type
5. Add comprehensive tests

### Configuration Customization
Key configuration areas for different use cases:
- **chunk_size/chunk_overlap**: Optimize based on document types
- **embedding models**: Balance accuracy vs. speed (MiniLM vs. larger models)
- **whisper_model**: Choose based on accuracy needs (tiny/base/small/medium/large)
- **vector_store.index_type**: Consider IndexIVFFlat for large datasets

### Performance Considerations
- Document processing: ~100 pages/minute for text documents
- Image OCR: ~10 images/minute (depends on image complexity)
- Audio transcription: ~5x real-time (model-dependent)
- Query response: <2 seconds for most queries
- Use GPU acceleration for embedding models when available

### Error Handling Patterns
- Graceful degradation for unsupported file types
- Fallback extraction methods (e.g., PyPDF2 after pdfplumber fails)
- Comprehensive logging with structured error information
- Batch processing continues on individual file failures

## Important Configuration Files

- `config/config.yaml`: Main system configuration
- `.env`: Environment variables for API keys and paths
- `requirements.txt`: Python dependencies including ML libraries
- `docker-compose.yml`: Multi-container setup with Ollama and optional ChromaDB/Redis

## Dependencies and External Services

### Required Services
- **Ollama**: Local LLM inference server (default: llama2 model)
- **Tesseract OCR**: Image text extraction

### Optional Services (Docker Compose)
- **ChromaDB**: Alternative vector store to FAISS
- **Redis**: Caching layer for improved performance

### Key Python Libraries
- **LangChain**: Text processing utilities
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **OpenAI Whisper**: Audio transcription
- **FastAPI**: API framework (when running in server mode)
