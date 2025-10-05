# Multi-stage build for multimodal RAG system
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Create data directories
RUN mkdir -p data/raw data/processed data/embeddings logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Create non-root user
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app
USER raguser

# Expose port for API (if using FastAPI)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.multimodal_rag.api"]
