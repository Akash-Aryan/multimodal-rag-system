# How to Run the Multimodal RAG API

## Issues Fixed ✅

The original bugs have been identified and fixed:

1. **Import errors** - Fixed dependency checks and graceful fallbacks
2. **Module path issues** - Removed complex imports that caused conflicts
3. **FastAPI deprecation warnings** - Updated to modern lifespan events
4. **Startup/shutdown issues** - Fixed server lifecycle management
5. **Windows Unicode errors** - Replaced emoji with plain text for Windows compatibility

## 🚀 Quick Start Methods

### Method 1: Super Easy (Windows) - Double-Click Method!

1. **Go to your project folder**: `C:\Users\akash\multimodal-rag-system`
2. **Double-click on**: `start_server.bat`
3. **A black window will open** showing:
   ```
   [SUCCESS] All dependencies loaded successfully
   [READY] API is ready!
   [URL] Access API at: http://localhost:8000
   ```
4. **Open your browser** and go to: http://localhost:8000/docs
5. **Done!** Your API is running!

### Method 2: Run Directly in VS Code

1. **Open VS Code** in this directory:
   ```bash
   code .
   ```

2. **Open the file**: `api_server.py`

3. **Press F5** or go to `Run > Start Debugging`
   - This will use the VS Code debugger configuration
   - Choose "Run RAG API Server" from the dropdown

4. **The server will start** and show:
   ```
   🚀 Starting Multimodal RAG API...
   📍 Access API at: http://localhost:8000
   📖 Documentation at: http://localhost:8000/docs
   ```

### Method 2: Terminal in VS Code

1. **Open integrated terminal** (`Ctrl + ` `)

2. **Run the server**:
   ```bash
   python api_server.py
   ```

### Method 3: With Auto-Reload (Development)

1. **In VS Code terminal**, run:
   ```bash
   uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📍 Access Your API

Once running, open these links:

- **🏠 Main API**: http://localhost:8000
- **📖 Interactive Docs**: http://localhost:8000/docs
- **❤️ Health Check**: http://localhost:8000/health
- **📊 System Stats**: http://localhost:8000/stats
- **💡 Examples**: http://localhost:8000/example-endpoints

## 🧪 Test the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Add Some Text
```bash
curl -X POST "http://localhost:8000/ingest-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample document about artificial intelligence and machine learning."}'
```

### 3. Query Your Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?", "top_k": 3}'
```

## 🔧 Troubleshooting

### If you get import errors:
```bash
pip install fastapi uvicorn sentence-transformers scikit-learn openai python-multipart
```

### If the server doesn't stay running:
- Check if another process is using port 8000
- Try changing the port: set environment variable `PORT=8001`
- Make sure you're not pressing Ctrl+C accidentally

### If embeddings don't work:
- The API will work with simple keyword matching
- For full functionality, install: `pip install sentence-transformers`

## 🎯 VS Code Debugging

The API is set up for easy debugging in VS Code:

1. **Set breakpoints** by clicking left of line numbers
2. **Use F5** to start debugging
3. **Variables panel** shows all values
4. **Debug console** for testing expressions

## 🌐 Features Available

✅ **Working without ML dependencies** - Basic keyword search  
✅ **Working with ML dependencies** - Full embedding-based search  
✅ **File upload** - Supports .txt files  
✅ **Interactive API docs** - Built-in Swagger UI  
✅ **Health monitoring** - Status and statistics endpoints  
✅ **CORS enabled** - Works with web frontends  

## 📚 Next Steps

1. **Try the interactive docs** at http://localhost:8000/docs
2. **Upload some text files** using the `/upload-file` endpoint
3. **Experiment with queries** to see semantic search in action
4. **Add your OpenAI API key** to `.env` file for AI-generated responses

## 💡 Pro Tips

- Use **F5 in VS Code** for the best development experience
- The **`/docs` endpoint** is perfect for testing without curl commands
- **Logs appear in VS Code terminal** when debugging
- **Auto-reload** restarts server when you save file changes
