"""
Lightweight FastAPI version of the multimodal RAG system.
This version removes heavy dependencies and works in VS Code.
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

# Global variables for the lightweight system
embedding_model = None
document_store = []  # In-memory storage (use database in production)

# Lightweight alternatives to heavy dependencies
DEPENDENCIES_AVAILABLE = False
try:
    import openai  # Use OpenAI API instead of local models
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
    print("[SUCCESS] All dependencies loaded successfully")
except ImportError as e:
    print(f"[WARNING] Some dependencies missing: {e}")
    print("[INFO] Install with: pip install openai sentence-transformers scikit-learn")
    print("[INFO] API will run with limited functionality")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the lightweight system."""
    global embedding_model
    
    print("[STARTUP] Starting Multimodal RAG API...")
    
    if DEPENDENCIES_AVAILABLE:
        try:
            # Use a smaller embedding model
            print("[LOADING] Loading embedding model...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[SUCCESS] Embedding model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            print("[INFO] API will run without embeddings")
    else:
        print("[WARNING] Running in limited mode - install dependencies for full functionality")
    
    print("[READY] API is ready!")
    print("[URL] Access API at: http://localhost:8000")
    print("[DOCS] Documentation at: http://localhost:8000/docs")
    print("[HEALTH] Health check at: http://localhost:8000/health")
    
    yield
    
    # Cleanup
    print("[SHUTDOWN] Shutting down API...")

app = FastAPI(
    title="Multimodal RAG API", 
    version="1.0.0", 
    description="Lightweight RAG system with document ingestion and querying",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class IngestRequest(BaseModel):
    text: str

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
    dependencies_available: bool
    embedding_model_loaded: bool

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Welcome to Multimodal RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Multimodal RAG API is running",
        dependencies_available=DEPENDENCIES_AVAILABLE,
        embedding_model_loaded=embedding_model is not None
    )

@app.post("/ingest-text")
async def ingest_text(request: IngestRequest):
    """Ingest plain text content."""
    if not DEPENDENCIES_AVAILABLE or embedding_model is None:
        # Store text without embeddings for basic functionality
        chunks = [request.text[i:i+1000] for i in range(0, len(request.text), 800)]
        
        for i, chunk in enumerate(chunks):
            document_store.append({
                "id": f"text_{len(document_store)}",
                "content": chunk,
                "embedding": None,  # No embedding without dependencies
                "metadata": {"type": "text", "chunk_id": i}
            })
        
        return {
            "message": f"Ingested {len(chunks)} text chunks (no embeddings - install dependencies for full functionality)",
            "chunks": len(chunks),
            "warning": "Limited functionality - install sentence-transformers for embeddings"
        }
    
    try:
        # Simple text chunking
        chunks = [request.text[i:i+1000] for i in range(0, len(request.text), 800)]
        
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
        
        return {"message": f"Ingested {len(chunks)} text chunks with embeddings", "chunks": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting text: {str(e)}")

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
            # For demo purposes - you'd use PyPDF2 or pdfplumber for real PDFs
            text_content = f"Demo PDF content from {file.filename}. In production, this would be extracted from the actual PDF."
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}. Supported: .txt, .pdf")
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Process the extracted text
        result = await ingest_text(IngestRequest(text=text_content))
        result["filename"] = file.filename
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document store."""
    try:
        if not document_store:
            raise HTTPException(status_code=400, detail="No documents ingested yet. Use /ingest-text or /upload-file first.")
        
        # If no embeddings available, use simple text matching
        if not DEPENDENCIES_AVAILABLE or embedding_model is None:
            # Simple keyword matching
            query_lower = request.query.lower()
            matches = []
            
            for doc in document_store:
                content_lower = doc["content"].lower()
                # Simple scoring based on keyword matches
                score = sum(1 for word in query_lower.split() if word in content_lower)
                if score > 0:
                    matches.append((score, doc))
            
            # Sort by score and get top-k
            matches.sort(key=lambda x: x[0], reverse=True)
            top_documents = matches[:request.top_k]
            
            if not top_documents:
                return QueryResponse(
                    response="No relevant documents found for your query.",
                    sources=[],
                    retrieved_count=0
                )
            
            # Create context from retrieved documents
            context = "\n\n".join([doc["content"] for _, doc in top_documents])
            answer = f"Based on simple keyword matching:\n\n{context[:500]}..."
            
            return QueryResponse(
                response=answer,
                sources=[
                    {
                        "content": doc["content"][:200] + "...",
                        "metadata": doc["metadata"],
                        "similarity": float(score) / 10  # Normalize score
                    }
                    for score, doc in top_documents
                ],
                retrieved_count=len(top_documents)
            )
        
        # Full embedding-based search
        # Generate query embedding
        query_embedding = embedding_model.encode([request.query])
        
        # Calculate similarities
        similarities = []
        for doc in document_store:
            if doc["embedding"] is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    doc["embedding"].reshape(1, -1)
                )[0][0]
                similarities.append((similarity, doc))
        
        if not similarities:
            raise HTTPException(status_code=400, detail="No documents with embeddings found")
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_documents = similarities[:request.top_k]
        
        # Create context from retrieved documents
        context = "\n\n".join([doc["content"] for _, doc in top_documents])
        
        # Generate response using OpenAI (or return context)
        if os.getenv("OPENAI_API_KEY"):
            try:
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
            except Exception as e:
                answer = f"OpenAI API error: {str(e)}\n\nContext-based answer: {context[:500]}..."
        else:
            answer = f"Context-based answer (set OPENAI_API_KEY for AI-generated responses):\n\n{context[:500]}..."
        
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
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    total_docs = len(document_store)
    docs_with_embeddings = sum(1 for doc in document_store if doc["embedding"] is not None)
    
    return {
        "total_documents": total_docs,
        "documents_with_embeddings": docs_with_embeddings,
        "embedding_model": "all-MiniLM-L6-v2" if embedding_model else "none",
        "dependencies_available": DEPENDENCIES_AVAILABLE,
        "api_version": "1.0.0",
        "status": "running"
    }

@app.delete("/clear")
async def clear_documents():
    """Clear all documents from memory."""
    global document_store
    count = len(document_store)
    document_store = []
    return {"message": f"Cleared {count} documents from memory"}

# Example endpoints for testing
@app.get("/example-endpoints")
async def example_endpoints():
    """Show example API usage."""
    return {
        "examples": {
            "1. Health Check": "GET /health",
            "2. Ingest Text": "POST /ingest-text with JSON: {'text': 'Your document content here'}",
            "3. Upload File": "POST /upload-file with form-data file",
            "4. Query": "POST /query with JSON: {'query': 'Your question here', 'top_k': 3}",
            "5. Get Stats": "GET /stats",
            "6. Clear Data": "DELETE /clear"
        },
        "note": "Visit /docs for interactive API documentation"
    }

if __name__ == "__main__":
    # This allows the script to be run directly
    print("[SERVER] Starting server...")
    print("[TIP] Open this file in VS Code and press F5 to run with debugger")
    
    port = int(os.getenv("PORT", 8000))
    
    # Use reload=True for development
    uvicorn.run(
        app,  # Pass the app directly, not a string
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for auto-reload during development
        log_level="info"
    )
