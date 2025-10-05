# Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system leveraging Large Language Models (LLMs) for **OFFLINE mode** that can ingest, index, and query diverse data formats including documents, images, and voice recordings within a unified semantic retrieval framework.

## 🎯 Project Overview

This system provides:

- **Multimodal Content Processing**: Handle text documents (PDF, DOCX, TXT, MD), images (JPG, PNG, etc.), and audio files (WAV, MP3, etc.)
- **Offline LLM Integration**: Works with local LLMs via Ollama for complete privacy and offline operation
- **Semantic Search**: Vector-based similarity search across all content types
- **Unified Query Interface**: Ask questions about your documents, images, and audio content in natural language
- **Flexible Architecture**: Modular design supporting different embedding models, vector stores, and LLMs

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │     Image       │    │     Audio       │
│   Processor     │    │   Processor     │    │   Processor     │
│  (PDF,DOCX,TXT) │    │ (JPG,PNG,etc.)  │    │ (WAV,MP3,etc.)  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │    Embedding Manager        │
                    │ (Text & Multimodal Models)  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Vector Store           │
                    │    (FAISS/Chroma)           │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Multimodal Retriever      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      LLM Manager            │
                    │   (Ollama Integration)      │
                    └─────────────────────────────┘
```

## 🚀 Features

### Data Ingestion
- **Text Documents**: PDF, DOCX, TXT, Markdown files
- **Images**: JPG, PNG, GIF with OCR text extraction
- **Audio**: WAV, MP3, M4A with speech-to-text transcription
- **Batch Processing**: Ingest entire directories of mixed content

### Processing Capabilities
- **Text Chunking**: Intelligent text splitting with overlap
- **OCR**: Extract text from images using Tesseract
- **Speech Recognition**: Audio transcription using OpenAI Whisper
- **Metadata Extraction**: File hashes, sizes, processing timestamps

### Search & Retrieval
- **Semantic Search**: Vector similarity search across all content types
- **Hybrid Queries**: Combine text, image, and audio queries
- **Configurable Results**: Adjust similarity thresholds and result counts
- **Source Attribution**: Full traceability to original files

### LLM Integration
- **Offline Operation**: Use local LLMs via Ollama
- **Context-Aware Responses**: RAG-enhanced generation with retrieved context
- **Configurable Models**: Support for different model sizes and parameters

## 📋 Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) for local LLM inference
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for image text extraction

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd multimodal-rag-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

#### Tesseract OCR
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

#### Ollama
1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull a model: `ollama pull llama2`
3. Verify installation: `ollama list`

### 5. Configuration

```bash
cp .env.example .env
# Edit .env with your specific paths and configuration
```

## 🔧 Configuration

The system uses YAML configuration files located in `config/config.yaml`. Key settings include:

```yaml
# LLM Configuration
llm:
  provider: "ollama"
  model_name: "llama2"
  temperature: 0.7
  max_tokens: 1024

# Vector Store
vector_store:
  provider: "faiss"
  dimension: 384
  persist_directory: "./data/embeddings"

# Document Processing
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from src.multimodal_rag import MultimodalRAG, RAGConfig

async def main():
    # Initialize the system
    config = RAGConfig.from_yaml("config/config.yaml")
    rag_system = MultimodalRAG(config)
    
    # Ingest documents
    await rag_system.ingest_file("path/to/document.pdf")
    await rag_system.ingest_file("path/to/image.jpg")
    await rag_system.ingest_file("path/to/audio.mp3")
    
    # Or ingest entire directory
    await rag_system.ingest_directory("data/raw")
    
    # Query the system
    result = await rag_system.query(
        "What are the main topics discussed in the documents?",
        query_type="text",
        top_k=5
    )
    
    print(f"Answer: {result['response']}")
    print(f"Sources: {len(result.get('sources', []))} documents")

# Run the async function
asyncio.run(main())
```

### CLI Usage

```bash
# Ingest documents
python scripts/ingest.py --directory data/raw

# Query the system
python scripts/query.py --query "What is discussed in the documents?" --top-k 5

# Check system status
python scripts/status.py
```

## 📖 Advanced Usage

### Custom Configuration

```python
from src.multimodal_rag import RAGConfig, MultimodalRAG

# Create custom configuration
config = RAGConfig(
    llm=LLMConfig(
        model_name="mistral",
        temperature=0.3
    ),
    vector_store=VectorStoreConfig(
        provider="chroma",
        dimension=768
    )
)

rag_system = MultimodalRAG(config)
```

### Batch Processing

```python
# Process multiple files
files = ["doc1.pdf", "image1.jpg", "audio1.wav"]
results = []

for file_path in files:
    result = await rag_system.ingest_file(file_path)
    results.append(result)

# Save index
await rag_system.save_index()
```

### Custom Queries

```python
# Text query
text_result = await rag_system.query(
    "Summarize the key findings",
    query_type="text",
    top_k=3
)

# Image-based query (using OCR text)
image_result = await rag_system.query(
    "path/to/query_image.jpg",
    query_type="image",
    top_k=5
)
```

## 🐳 Docker Deployment

```bash
# Build the image
docker-compose build

# Run the system
docker-compose up -d

# Access the API
curl http://localhost:8000/query -X POST -H "Content-Type: application/json" -d '{"query": "What is in the documents?", "top_k": 5}'
```

## 📁 Project Structure

```
multimodal-rag-system/
├── src/multimodal_rag/           # Core package
│   ├── ingestion/                # Data ingestion modules
│   │   ├── document_processor.py # Text document processing
│   │   ├── image_processor.py    # Image processing & OCR
│   │   └── audio_processor.py    # Audio transcription
│   ├── indexing/                 # Indexing and embeddings
│   │   ├── vector_store.py       # Vector database interface
│   │   └── embeddings.py         # Embedding generation
│   ├── retrieval/                # Search and retrieval
│   │   └── retriever.py          # Multimodal retriever
│   ├── models/                   # LLM integration
│   │   └── llm_manager.py        # LLM interface
│   ├── config.py                 # Configuration management
│   └── core.py                   # Main RAG system
├── config/                       # Configuration files
│   └── config.yaml               # Default configuration
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── examples/                     # Example usage
├── tests/                        # Test suite
├── data/                         # Data directories
│   ├── raw/                      # Raw input files
│   ├── processed/                # Processed data
│   └── embeddings/               # Vector embeddings
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # Multi-container setup
└── README.md                     # This file
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_ingestion.py
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=src tests/
```

## 🔄 API Reference

### Core Methods

#### `MultimodalRAG.ingest_file(file_path: str) -> Dict`
Ingest a single file into the system.

#### `MultimodalRAG.ingest_directory(directory_path: str) -> List[Dict]`
Ingest all supported files in a directory.

#### `MultimodalRAG.query(query: str, query_type: str = "text", top_k: int = 5) -> Dict`
Query the RAG system and get AI-generated responses.

#### `MultimodalRAG.get_system_stats() -> Dict`
Get system statistics and status information.

## 🛠️ Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Adding New Data Types

1. Create a new processor in `src/multimodal_rag/ingestion/`
2. Update the supported formats in `config.py`
3. Add processing logic to `core.py`
4. Update the embedding manager for new content types
5. Add tests for the new functionality

## 📊 Performance

### Benchmarks
- **Text Processing**: ~100 pages/minute
- **Image OCR**: ~10 images/minute
- **Audio Transcription**: ~5x real-time (depends on Whisper model)
- **Query Response**: <2 seconds for most queries

### Optimization Tips
- Use GPU acceleration for embedding models
- Adjust chunk sizes based on your content
- Consider using larger Whisper models for better accuracy
- Optimize FAISS index type for your use case

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [LangChain](https://github.com/langchain-ai/langchain) for text processing utilities

## 📬 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Join our community discussions

---

**Built with ❤️ for the open-source community**
