# Deployment Guide for Multimodal RAG System

## Why Vercel Won't Work

❌ **Vercel is not suitable for this project because:**
- 50MB deployment size limit (this project needs >2GB for ML models)
- 60-second function timeout (processing takes minutes)
- No persistent storage for embeddings
- Cannot install system dependencies (Tesseract, Ollama)
- Serverless functions can't handle heavy ML workloads

## Recommended Deployment Options

### 🚀 Option 1: Railway (Easiest)

**Pros:** Simple deployment, good for prototypes, persistent storage
**Cons:** Can be expensive for heavy usage

#### Steps:
1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   # or
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. **Login and deploy:**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set environment variables in Railway dashboard:**
   - `OPENAI_API_KEY` (if using OpenAI instead of Ollama)
   - `LOG_LEVEL=INFO`

#### Cost: ~$20-50/month depending on usage

---

### 🏭 Option 2: Google Cloud Run (Most Scalable)

**Pros:** Pay-per-use, scales to zero, handles heavy workloads
**Cons:** Requires GCP setup, more complex

#### Steps:
1. **Install Google Cloud CLI:**
   ```bash
   # Windows (PowerShell)
   (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
   & $env:Temp\GoogleCloudSDKInstaller.exe
   ```

2. **Setup and deploy:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   
   # Build and push to Container Registry
   docker build -t gcr.io/YOUR_PROJECT_ID/multimodal-rag .
   docker push gcr.io/YOUR_PROJECT_ID/multimodal-rag
   
   # Deploy to Cloud Run
   gcloud run deploy multimodal-rag \
     --image gcr.io/YOUR_PROJECT_ID/multimodal-rag \
     --platform managed \
     --region us-central1 \
     --memory 8Gi \
     --cpu 4 \
     --timeout 3600 \
     --allow-unauthenticated
   ```

#### Cost: ~$10-30/month with auto-scaling

---

### 💻 Option 3: DigitalOcean App Platform

**Pros:** Simple Docker deployment, reasonable pricing
**Cons:** Less flexibility than GCP

#### Steps:
1. **Create `app.yaml`:**
   ```yaml
   name: multimodal-rag
   services:
   - name: web
     source_dir: /
     github:
       repo: your-username/multimodal-rag-system
       branch: main
     run_command: python -m src.multimodal_rag.api_lightweight
     environment_slug: python
     instance_count: 1
     instance_size_slug: professional-xs
     http_port: 8000
     env:
     - key: PYTHONPATH
       value: /app
   ```

2. **Deploy via DigitalOcean dashboard or CLI**

#### Cost: ~$25-50/month

---

### 🛠 Option 4: Lightweight Version (Works on More Platforms)

For platforms like Render, Heroku, or even modified Vercel approach:

#### Create lightweight requirements:
```bash
# Create requirements-light.txt
cat > requirements-light.txt << EOF
fastapi>=0.100.0
uvicorn>=0.20.0
sentence-transformers>=2.2.0
scikit-learn>=1.1.0
numpy>=1.21.0
openai>=1.0.0
python-multipart>=0.0.5
EOF
```

#### Deploy to Render:
1. Connect your GitHub repo to Render
2. Use `src/multimodal_rag/api_lightweight.py` as entry point
3. Set build command: `pip install -r requirements-light.txt`
4. Set start command: `python src/multimodal_rag/api_lightweight.py`

---

## Quick Start Instructions

### For Railway (Recommended):

```bash
# 1. Create Railway account at railway.app
# 2. Install CLI
npm install -g @railway/cli

# 3. Deploy
railway login
railway init
railway up

# 4. Your API will be available at the provided Railway URL
```

### For local testing of lightweight version:

```bash
# Install lightweight dependencies
pip install fastapi uvicorn sentence-transformers scikit-learn openai python-multipart

# Run the API
python src/multimodal_rag/api_lightweight.py

# Test endpoints:
# http://localhost:8000/docs - Interactive API documentation
# POST http://localhost:8000/ingest-text - Add text content
# POST http://localhost:8000/query - Query documents
```

## Environment Variables Needed

```bash
# Required for OpenAI integration (lightweight version)
OPENAI_API_KEY=your_openai_api_key_here

# Optional
LOG_LEVEL=INFO
PORT=8000

# For full version with external vector DB
PINECONE_API_KEY=your_pinecone_key_here
CHROMADB_HOST=your_chromadb_host
```

## Testing Your Deployment

Once deployed, test with:

```bash
# Health check
curl https://your-app-url.com/health

# Upload text
curl -X POST "https://your-app-url.com/ingest-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample document for testing."}'

# Query
curl -X POST "https://your-app-url.com/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?", "top_k": 3}'
```

## Production Considerations

1. **Use external vector database** (Pinecone, Weaviate, or Qdrant)
2. **Add authentication** for API endpoints
3. **Implement rate limiting**
4. **Add monitoring and logging**
5. **Use persistent storage** for uploaded files
6. **Consider GPU instances** for better performance

Would you like me to help you deploy to any specific platform?
