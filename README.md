# Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system leveraging Large Language Models (LLMs) for **OFFLINE mode** that can ingest, index, and query diverse data formats including documents, images, and voice recordings within a unified semantic retrieval framework.

## 🎯 Project Overview

This system provides:

- **Multimodal Content Processing**: Handle text documents (PDF, DOCX, TXT, MD), images (JPG, PNG, etc.), and audio files (WAV, MP3, etc.)
- **Offline LLM Integration**: Works with local LLMs via Ollama for complete privacy and offline operation
- **Semantic Search**: Vector-based similarity search across all content types
- **Unified Query Interface**: Ask questions about your documents, images, and audio content in natural language
- **Flexible Architecture**: Modular design supporting different embedding models, vector stores, and LLMs



## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                                  │
│                         (React Native + UI)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Unified Search  │  │  Upload Manager  │  │  Result Display  │          │
│  │   & Chat UI      │  │  (Doc/Img/Audio) │  │   with Citations │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                      │                     │
│           └─────────────────────┼──────────────────────┘                     │
│                                 │                                            │
└─────────────────────────────────┼────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
│                    (Use Cases & Orchestration)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐   │
│  │  Ingestion Use Cases │  │   Query Use Cases    │  │ Security Cases  │   │
│  ├──────────────────────┤  ├──────────────────────┤  ├─────────────────┤   │
│  │ • DocumentIngestion  │  │ • SemanticSearch     │  │ • Authentication│   │
│  │ • ImageIngestion     │  │ • CrossModalRetrieval│  │ • Authorization │   │
│  │ • AudioIngestion     │  │ • AnswerGeneration   │  │ • Encryption    │   │
│  │ • BatchProcessing    │  │ • CitationLinking    │  │ • AuditLogging  │   │
│  └──────────┬───────────┘  └──────────┬───────────┘  └────────┬────────┘   │
│             │                         │                        │             │
└─────────────┼─────────────────────────┼────────────────────────┼─────────────┘
              │                         │                        │
┌─────────────▼─────────────────────────▼────────────────────────▼─────────────┐
│                           DOMAIN LAYER                                        │
│                   (Core Business Logic - Framework Agnostic)                  │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        DOMAIN SERVICES                                 │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                         │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                   │  │
│  │  │ EmbeddingService    │    │ TranscriptionService│                   │  │
│  │  │ • Text Embedding    │    │ • Audio-to-Text     │                   │  │
│  │  │ • Image Embedding   │    │ • Whisper Model     │                   │  │
│  │  │ • Audio Embedding   │    │ • Timestamp Sync    │                   │  │
│  │  └─────────────────────┘    └─────────────────────┘                   │  │
│  │                                                                         │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                   │  │
│  │  │ DocumentParserService│    │ ImageAnalysisService│                   │  │
│  │  │ • PDF Extraction    │    │ • OCR (Tesseract)   │                   │  │
│  │  │ • DOCX Parsing      │    │ • Image Captioning  │                   │  │
│  │  │ • Metadata Extract  │    │ • Visual Features   │                   │  │
│  │  └─────────────────────┘    └─────────────────────┘                   │  │
│  │                                                                         │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                   │  │
│  │  │ LLMInferenceService │    │  CitationService    │                   │  │
│  │  │ • Answer Generation │    │ • Source Tracking   │                   │  │
│  │  │ • Context Grounding │    │ • Citation Linking  │                   │  │
│  │  │ • Quantized Models  │    │ • Provenance Chain  │                   │  │
│  │  └─────────────────────┘    └─────────────────────┘                   │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    DOMAIN ENTITIES & VALUE OBJECTS                     │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  Document │ Image │ Audio │ Query │ Answer │ Citation │ Embedding     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      DOMAIN INTERFACES (PORTS)                         │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  IVectorStore │ IDocumentRepo │ IMLModelRepo │ IEncryption │ IAuth    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼─────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                                  │
│                  (Technical Implementation & Adapters)                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       INGESTION PIPELINE                                ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                          ││
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   ││
│  │   │   Document      │    │     Image       │    │     Audio       │   ││
│  │   │   Processor     │    │   Processor     │    │   Processor     │   ││
│  │   │ (PDF,DOCX,TXT)  │    │ (JPG,PNG,WEBP)  │    │ (WAV,MP3,M4A)   │   ││
│  │   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘   ││
│  │            │                      │                       │            ││
│  │            │  ┌───────────────────┴───────────────────┐   │            ││
│  │            └─▶│      Content Extraction Layer         │◀──┘            ││
│  │               │  • Text: PyPDF2, python-docx          │                ││
│  │               │  • Image: Pillow, Tesseract OCR       │                ││
│  │               │  • Audio: Whisper (whisper-tiny)      │                ││
│  │               └────────────────┬──────────────────────┘                ││
│  │                                │                                       ││
│  │               ┌────────────────▼──────────────────────┐                ││
│  │               │      Metadata Enrichment              │                ││
│  │               │  • Timestamps, File Info, User ID     │                ││
│  │               │  • Content Type, Size, Checksum       │                ││
│  │               └────────────────┬──────────────────────┘                ││
│  │                                │                                       ││
│  └────────────────────────────────┼───────────────────────────────────────┘│
│                                   │                                        │
│  ┌────────────────────────────────▼───────────────────────────────────────┐│
│  │                      EMBEDDING PIPELINE                                ││
│  ├────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │    ┌────────────────────────────────────────────────────────────┐     ││
│  │    │            Embedding Manager (Unified)                     │     ││
│  │    ├────────────────────────────────────────────────────────────┤     ││
│  │    │                                                             │     ││
│  │    │  ┌──────────────────┐  ┌──────────────────┐               │     ││
│  │    │  │  Text Embedder   │  │  Image Embedder  │               │     ││
│  │    │  │  • all-MiniLM-L6 │  │  • CLIP (ViT-B)  │               │     ││
│  │    │  │  • 384 dimensions│  │  • 512 dimensions│               │     ││
│  │    │  │  Size: ~80MB     │  │  Size: ~150MB    │               │     ││
│  │    │  └──────────────────┘  └──────────────────┘               │     ││
│  │    │                                                             │     ││
│  │    │  ┌──────────────────┐  ┌──────────────────┐               │     ││
│  │    │  │  Audio Embedder  │  │ Cross-Modal Proj │               │     ││
│  │    │  │  • Wav2Vec2-base │  │ • Unified 512-dim│               │     ││
│  │    │  │  • 768→512 proj  │  │ • Similarity Map │               │     ││
│  │    │  │  Size: ~95MB     │  │ • Fusion Layer   │               │     ││
│  │    │  └──────────────────┘  └──────────────────┘               │     ││
│  │    │                                                             │     ││
│  │    └─────────────────────────┬───────────────────────────────────     ││
│  │                              │                                         ││
│  └──────────────────────────────┼─────────────────────────────────────────┘│
│                                 │                                          │
│  ┌──────────────────────────────▼─────────────────────────────────────────┐│
│  │                       STORAGE LAYER                                    ││
│  ├────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │                    Vector Store (FAISS)                         │  ││
│  │  ├─────────────────────────────────────────────────────────────────┤  ││
│  │  │  • IndexFlatIP: Exact search for <10K vectors                   │  ││
│  │  │  • IndexIVFFlat: Approximate search for 10K-1M vectors          │  ││
│  │  │  • Quantization: PQ/SQ for compression                          │  ││
│  │  │  • Partitioning: By modality & content type                     │  ││
│  │  │  • Persistence: Serialized to encrypted filesystem              │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │              Metadata Store (SQLite/Realm)                      │  ││
│  │  ├─────────────────────────────────────────────────────────────────┤  ││
│  │  │  Tables:                                                         │  ││
│  │  │  • documents(id, type, path, created_at, user_id, checksum)     │  ││
│  │  │  • embeddings(id, doc_id, vector_id, modality)                  │  ││
│  │  │  • queries(id, text, user_id, timestamp, result_count)          │  ││
│  │  │  • citations(id, answer_id, source_id, snippet, position)       │  ││
│  │  │  • audit_logs(id, user_id, action, timestamp, details)          │  ││
│  │  │  • Encryption: SQLCipher for database-level encryption          │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │              File System Storage (Encrypted)                    │  ││
│  │  ├─────────────────────────────────────────────────────────────────┤  ││
│  │  │  • Raw Files: AES-256-GCM encrypted blobs                        │  ││
│  │  │  • Cache: Query results, thumbnails, processed data             │  ││
│  │  │  • Models: Quantized ML models (GGUF/ONNX format)               │  ││
│  │  │  • Keys: Hardware-backed Keystore/Keychain                       │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       RETRIEVAL PIPELINE                                ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                          ││
│  │   ┌──────────────────────────────────────────────────────────────┐     ││
│  │   │              Query Processor                                 │     ││
│  │   │  • Intent Classification (search vs chat)                    │     ││
│  │   │  • Query Expansion & Reformulation                           │     ││
│  │   │  • Modality Detection (text/image/audio input)               │     ││
│  │   └─────────────────────────┬────────────────────────────────────┘     ││
│  │                             │                                          ││
│  │   ┌─────────────────────────▼────────────────────────────────────┐     ││
│  │   │          Multimodal Retriever                                │     ││
│  │   ├──────────────────────────────────────────────────────────────┤     ││
│  │   │  • Hybrid Search: Vector (semantic) + Keyword (BM25)         │     ││
│  │   │  • Cross-Modal Matching: Text→Image, Audio→Text, etc.        │     ││
│  │   │  • Reranking: MMR (Maximal Marginal Relevance)               │     ││
│  │   │  • Filtering: RBAC, date range, content type                 │     ││
│  │   │  • Top-K Selection: Configurable (default: 10-20 results)    │     ││
│  │   └─────────────────────────┬────────────────────────────────────┘     ││
│  │                             │                                          ││
│  └─────────────────────────────┼──────────────────────────────────────────┘│
│                                │                                           │
│  ┌─────────────────────────────▼──────────────────────────────────────────┐│
│  │                    GENERATION PIPELINE                                 ││
│  ├────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐ ││
│  │   │               Context Preparation                               │ ││
│  │   │  • Retrieved chunks formatting                                  │ ││
│  │   │  • Source citation tagging                                      │ ││
│  │   │  • Context window optimization (<2048 tokens)                   │ ││
│  │   └──────────────────────────┬──────────────────────────────────────┘ ││
│  │                              │                                        ││
│  │   ┌──────────────────────────▼──────────────────────────────────────┐ ││
│  │   │            LLM Engine (Quantized)                               │ ││
│  │   ├─────────────────────────────────────────────────────────────────┤ ││
│  │   │  Options (choose one):                                          │ ││
│  │   │  • Llama 3.2 1B (INT4 quantized, ~500MB)                        │ ││
│  │   │  • Gemma 2B (INT4 quantized, ~450MB)                            │ ││
│  │   │  • Phi-3 Mini (INT8 quantized, ~480MB)                          │ ││
│  │   │                                                                  │ ││
│  │   │  Inference: ONNX Runtime / TensorFlow Lite                      │ ││
│  │   │  Prompt Template: RAG-specific with citation instructions       │ ││
│  │   │  Generation: Greedy/Sampling with temperature control           │ ││
│  │   └──────────────────────────┬──────────────────────────────────────┘ ││
│  │                              │                                        ││
│  │   ┌──────────────────────────▼──────────────────────────────────────┐ ││
│  │   │         Answer Post-Processing                                  │ ││
│  │   │  • Citation extraction & linking                                │ ││
│  │   │  • Confidence scoring                                           │ ││
│  │   │  • Grounding verification (answer vs sources)                   │ ││
│  │   │  • Response formatting with markdown                            │ ││
│  │   └─────────────────────────────────────────────────────────────────┘ ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      SECURITY LAYER                                     ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                          ││
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     ││
│  │  │ Authentication   │  │  Authorization   │  │   Encryption     │     ││
│  │  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤     ││
│  │  │ • NTRO SSO       │  │ • RBAC Engine    │  │ • AES-256-GCM    │     ││
│  │  │ • Biometric Auth │  │ • Least Privilege│  │ • Hardware Keys  │     ││
│  │  │ • MFA (TOTP)     │  │ • Role Hierarchy │  │ • Key Rotation   │     ││
│  │  │ • Session Mgmt   │  │ • Access Logging │  │ • Secure Enclave │     ││
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘     ││
│  │                                                                          ││
│  │  ┌───────────────────────────────────────────────────────────────────┐ ││
│  │  │                    Audit & Monitoring                             │ ││
│  │  │  • All data access logged (who, what, when, where)               │ ││
│  │  │  • Tamper detection for models and data                          │ ││
│  │  │  • Anomaly detection in query patterns                           │ ││
│  │  │  • Compliance reporting (STQC, ISO 27001)                        │ ││
│  │  └───────────────────────────────────────────────────────────────────┘ ││
│  │                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Core Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  1️⃣ INGEST                  2️⃣ INDEX                   3️⃣ RETRIEVE           │
│  ┌──────────┐              ┌──────────┐              ┌──────────┐          │
│  │ Upload   │──Parse──────▶│ Embed    │──Store──────▶│ Query    │          │
│  │ Document │              │ Content  │              │ Vectors  │          │
│  └──────────┘              └──────────┘              └──────────┘          │
│       │                         │                         │                 │
│       │                         │                         │                 │
│       └─────────────────────────┴─────────────────────────┘                 │
│                                 │                                           │
│                                 ▼                                           │
│                          4️⃣ GROUND & CITE                                   │
│                          ┌──────────┐                                       │
│                          │ Generate │                                       │
│                          │ Answer + │                                       │
│                          │ Sources  │                                       │
│                          └──────────┘                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Technology Stack

### **Frontend/Mobile**
- **React Native** (0.72+): Cross-platform UI
- **React Navigation**: Screen routing
- **Native Modules**: Camera, File Picker, Audio Recorder
- **Async Storage**: Lightweight key-value cache

### **ML/AI Stack**
- **Embedding Models**:
  - Text: `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
  - Image: `openai/clip-vit-base-patch32` (~150MB)
  - Audio: `facebook/wav2vec2-base` (~95MB)
  
- **LLM Engine**: 
  - Quantized Llama 3.2 1B (INT4, ~500MB) OR
  - Quantized Gemma 2B (INT4, ~450MB)
  
- **Speech-to-Text**: `openai/whisper-tiny` (~150MB)

- **Inference Engines**:
  - TensorFlow Lite (mobile-optimized)
  - ONNX Runtime (cross-platform)

### **Storage**
- **Vector Database**: FAISS (Facebook AI Similarity Search)
  - Supports both exact and approximate search
  - Offline-capable with file-based persistence
  
- **Metadata Database**: SQLite with SQLCipher (encrypted)
- **File Storage**: AES-256-GCM encrypted filesystem
- **Key Management**: iOS Keychain / Android Keystore

### **Security**
- **Encryption**: AES-256-GCM with hardware-backed keys
- **Authentication**: NTRO SSO + Biometric (Face ID/Touch ID/Fingerprint)
- **Authorization**: Role-Based Access Control (RBAC)
- **Audit**: Complete activity logging with tamper detection

### **Processing Libraries**
- **Document**: PyPDF2, python-docx, BeautifulSoup
- **Image**: Pillow, Tesseract OCR
- **Audio**: librosa, pydub, ffmpeg

---

## 🔐 Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🔒 Authentication Layer                                     │
│     ├─ NTRO SSO Integration (SAML 2.0)                      │
│     ├─ Multi-Factor Authentication (TOTP + Biometric)       │
│     └─ Session Management (JWT with refresh tokens)         │
│                                                              │
│  🛡️  Authorization Layer                                     │
│     ├─ Role-Based Access Control (Admin/Analyst/Viewer)    │
│     ├─ Resource-Level Permissions                           │
│     └─ Least Privilege Enforcement                          │
│                                                              │
│  🔐 Data Protection Layer                                    │
│     ├─ At Rest: AES-256-GCM encryption                      │
│     ├─ In Transit: TLS 1.3 (for sync operations)            │
│     ├─ In Memory: Secure memory wiping after use            │
│     └─ Key Management: Hardware-backed secure storage       │
│                                                              │
│  📋 Audit & Compliance Layer                                │
│     ├─ Complete activity logging                            │
│     ├─ Tamper detection (checksums, signatures)             │
│     ├─ Compliance reports (STQC, ISO 27001)                 │
│     └─ Data retention policies                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Performance Optimization Strategies

### **Model Optimization**
- INT4/INT8 quantization for all models
- GGUF format for efficient storage and loading
- Lazy loading: Load models on-demand
- Model pruning: Remove redundant parameters

### **Vector Search Optimization**
- Use FAISS IndexIVFFlat for datasets > 10K vectors
- Product Quantization (PQ) for compression (8-bit)
- Partitioning by modality and content type
- Incremental indexing for new documents

### **Caching Strategy**
- Query result cache (LRU, 100 entries)
- Embedding cache for frequently accessed documents
- Thumbnail cache for images
- Preprocessed audio chunks

### **Resource Management**
- Background processing for indexing (not blocking UI)
- Batch processing with configurable chunk sizes
- Memory pooling for embeddings
- Automatic garbage collection for old cache

---

## 🎯 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Query Response Time** | < 2 seconds | End-to-end (input to display) |
| **Indexing Speed** | 100 docs/min | Background processing |
| **Accuracy** | 95%+ precision | Retrieval relevance |
| **RAM Usage** | < 2 GB | Active usage during query |
| **Storage Efficiency** | < 500MB per 10K docs | Including vectors + metadata |
| **Battery Consumption** | < 10% per hour | Active usage |
| **Model Footprint** | < 850 MB total | All models combined |
| **Cold Start** | < 5 seconds | App launch to ready state |

---

## 🏛️ Clean Architecture Compliance

### **Dependency Rule**
```
Presentation → Application → Domain → Infrastructure
     ↓              ↓            ↓            ↓
    UI          Use Cases    Business      Tech Details
                              Logic        & Frameworks
```

**Key Principle**: Dependencies point inward. Core domain is framework-agnostic.

### **Layer Responsibilities**

| Layer | Responsibility | No Dependencies On |
|-------|---------------|-------------------|
| **Presentation** | UI, user interaction | Application, Domain, Infrastructure |
| **Application** | Use case orchestration | Domain (interfaces only) |
| **Domain** | Business logic, entities | Any external frameworks |
| **Infrastructure** | Technical implementation | None (implements domain interfaces) |

---

## 🚀 Getting Started

### **Phase 1: Core Pipeline (Weeks 1-4)**
```bash
# 1. Document ingestion
mindra ingest documents/*.pdf

# 2. Vector indexing
mindra index --modality text

# 3. Query testing
mindra query "Find documents about satellite imagery"
```

### **Phase 2: Multimodal Support (Weeks 5-8)**
```bash
# Add image processing
mindra ingest images/*.jpg --with-ocr

# Add audio transcription
mindra ingest audio/*.mp3 --transc

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
