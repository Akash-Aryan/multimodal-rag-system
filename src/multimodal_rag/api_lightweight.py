"""
Lightweight FastAPI version of the multimodal RAG system.
This version removes heavy dependencies and uses cloud services.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import uvicorn
import os
import tempfile
from pathlib import Path
import asyncio

# Global variables for the lightweight system
embedding_model = None
document_store = []  # In-memory storage (use database in production)

# Lightweight alternatives to heavy dependencies
try:
    import openai  # Use OpenAI API instead of local models
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install openai sentence-transformers scikit-learn")
    DEPENDENCIES_AVAILABLE = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the lightweight system."""
    global embedding_model
    
    if DEPENDENCIES_AVAILABLE:
        try:
            # Use a smaller embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
    
    yield
    
    # Cleanup
    print("🔄 Shutting down API...")

app = FastAPI(title="Multimodal RAG API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    query_type: Optional[str] = "text"

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[dict]] = []
    retrieved_count: int

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Multimodal RAG API is running"
    )

@app.post("/ingest-text")
async def ingest_text(text: str):
    """Ingest plain text content."""
    if not DEPENDENCIES_AVAILABLE or embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not available. Install dependencies: pip install sentence-transformers")
    
    try:
        # Simple text chunking
        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
        
        # Generate embeddings
        embeddings = embedding_model.encode(chunks)
        
        # Store in memory (use vector DB in production)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            document_store.append({
                "id": f"text_{len(document_store)}",
                "content": chunk,
                "embedding": embedding,
                "metadata": {"type": "text", "chunk_id": i}
            })
        
        return {"message": f"Ingested {len(chunks)} text chunks", "chunks": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract text based on file type
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension == '.txt':
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        elif file_extension == '.pdf':
            # Simplified PDF extraction (you'd use PyPDF2 or pdfplumber)
            text_content = "PDF content extraction not implemented in lightweight version"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Process the extracted text
        result = await ingest_text(text_content)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document store."""
    if not DEPENDENCIES_AVAILABLE or embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not available. Install dependencies: pip install sentence-transformers")
    
    try:
        if not document_store:
            raise HTTPException(status_code=400, detail="No documents ingested yet")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([request.query])
        
        # Calculate similarities
        similarities = []
        for doc in document_store:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc["embedding"].reshape(1, -1)
            )[0][0]
            similarities.append((similarity, doc))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_documents = similarities[:request.top_k]
        
        # Create context from retrieved documents
        context = "\n\n".join([doc["content"] for _, doc in top_documents])
        
        # Generate response using OpenAI (or return context)
        if os.getenv("OPENAI_API_KEY"):
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.query}"}
                ],
                max_tokens=500
            )
            answer = response.choices[0].message.content
        else:
            answer = f"Context-based answer: {context[:500]}..."
        
        return QueryResponse(
            response=answer,
            sources=[
                {
                    "content": doc["content"][:200] + "...",
                    "metadata": doc["metadata"],
                    "similarity": float(sim)
                }
                for sim, doc in top_documents
            ],
            retrieved_count=len(top_documents)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "total_documents": len(document_store),
        "embedding_model": "all-MiniLM-L6-v2",
        "api_version": "1.0.0",
        "status": "running"
    }

@app.delete("/clear")
async def clear_documents():
    """Clear all documents from memory."""
    global document_store
    document_store = []
    return {"message": "All documents cleared"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api_lightweight:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
