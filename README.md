# Local PDF RAG System

A **Local Retrieval-Augmented Generation (RAG)** application that enables you to chat with your PDF documents using a completely local setup. No data leaves your machine.

## Features

- 📄 **PDF Processing**: Extract and chunk text from PDF documents
- 🔍 **Vector Search**: Semantic search using ChromaDB and SentenceTransformers
- 🤖 **Local LLM**: Answer generation using Ollama (Mistral model)
- 🚀 **FastAPI**: REST API for easy integration
- 🐳 **Docker**: Containerized deployment with docker-compose
- 🧪 **Testing**: Comprehensive test suite with unittest

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd pdf-arg

# Start the services
docker-compose up --build

# The API will be available at http://localhost:8000

# Process your first PDF (required before querying)
python -c "from app.script import process_pdf; process_pdf()"
```

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama separately
# Download from https://ollama.ai
ollama pull phi

# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Process your first PDF (required before querying)
python -c "from app.script import process_pdf; process_pdf()"
```

## Usage

### Step 1: Embed Your PDF Documents (Required First Step)

**Before you can query any documents, you must first process and embed them into the vector database.**

```python
from app.script import process_pdf

# Process the included sample PDF
process_pdf()  # Uses default: "app/files/attention.pdf"

# OR process your own PDF file
process_pdf("path/to/your/document.pdf")
```

This step will:
1. Extract text from the PDF
2. Split it into chunks
3. Generate embeddings
4. Store everything in ChromaDB

### Step 2: Query Your Documents

Only after embedding your PDFs can you ask questions about them:

```python
from app.script import query_pdf

# Ask questions using the default query
query_pdf()  # Default: "What is attention mechanism?"

# Ask your own questions
query_pdf("What are the key findings?")
query_pdf("Summarize the methodology", top_k=3)  # Use fewer chunks
```

### Step 3: Check Your Database

```python
from app.script import collection_count

# See how many document chunks are stored
collection_count()
```

### Complete Workflow Example

```python
# 1. First, embed a PDF (required)
from app.script import process_pdf, collection_count, query_pdf

process_pdf("app/files/attention.pdf")  # Process the sample PDF
collection_count()  # Check that documents were added

# 2. Now you can query
query_pdf("What is the transformer architecture?")
query_pdf("Who are the authors of this paper?")
```

### API Endpoints
- `GET /`: Health check
- Additional endpoints can be added to `app/main.py`

## Project Structure

```
pdf-arg/
├── app/
│   ├── main.py              # FastAPI application
│   ├── script.py            # Processing and query scripts
│   ├── services/
│   │   ├── pdf_reader.py    # PDF text extraction
│   │   ├── chunker.py       # Text chunking with overlap
│   │   ├── embedder.py      # SentenceTransformers embeddings
│   │   ├── chroma_store.py  # ChromaDB vector operations
│   │   ├── llm_client.py    # Ollama LLM interface
│   │   └── rag_pipeline.py  # Complete RAG workflow
│   └── tests/               # Unit tests
├── data/
│   ├── chroma/             # ChromaDB storage (created automatically)
│   └── ollama/             # Ollama models (Docker volume)
├── requirements.txt        # Python dependencies
├── dockerfile             # API container image
└── docker-compose.yml     # Full stack orchestration
```

## Architecture Flow
```
               ┌────────────┐
               │   PDF(s)   │
               └─────┬──────┘
                     │
           ┌─────────▼─────────┐
           │   PDF Reader      │
           └────────┬──────────┘
                     │
           ┌─────────▼─────────┐
           │     Chunker       │
           │  -> list of dicts │
           └────────┬──────────┘
                     │
           ┌─────────▼─────────────┐
           │   Embedder            │
           │ (SentenceTransformers)│
           └────────┬──────────────┘
                     │
           ┌─────────▼─────────┐
           │  Chroma Vector DB │
           └───────────────────┘

User Query ───────┐
                   ▼
             ┌────────────────────────┐
             │   Retriever            │
             │  (query → top_k chunks)│
             └──────┬─────────────────┘
                    │
                    ▼
             ┌──────────────────┐
             │     LLM          │
             │ (Ollama/Mistral) │
             └──────┬───────────┘
                    │
                    ▼
                 Answer
```

## Development

### Running Tests
```bash
# Run all tests
python -m unittest discover app/tests/

```

### Configuration

The application can be configured through environment variables:

- `OLLAMA_HOST`: Ollama service hostname (default: `localhost`)
- `OLLAMA_PORT`: Ollama service port (default: `11434`)
- `PYTHONUNBUFFERED`: Set to `1` for immediate output

### Data Storage

- **ChromaDB**: Vector embeddings stored in `./data/chroma/`
- **Ollama Models**: Downloaded models stored in `./data/ollama/`
- **PDFs**: Sample PDFs can be placed in `./app/files/`

## Technology Stack

- **Backend**: Python 3.11, FastAPI, uvicorn
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (multi-qa-MiniLM-L6-cos-v1)
- **LLM**: Ollama with Mistral model
- **PDF Processing**: PyPDF2
- **Containerization**: Docker, docker-compose
- **Testing**: unittest, Rich (for output formatting)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python -m unittest discover app/tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Troubleshooting

### Common Issues

**Ollama Connection Error**
- Ensure Ollama is running: `docker-compose logs ollama`
- Check if Mistral model is downloaded: `ollama list`
- Verify network connectivity between containers

**ChromaDB Permission Issues**
- Ensure `./data/chroma/` directory has proper permissions
- On Linux/macOS: `chmod -R 755 ./data/`

**PDF Processing Fails**
- Check if PDF is readable and not encrypted
- Verify file path is correct
- Ensure PyPDF2 can handle the PDF format

### Performance Tips

- **Chunk Size**: Adjust `max_chars` in chunking for better retrieval (default: 1000)
- **Embedding Model**: Consider larger models for better semantic understanding
- **LLM Model**: Try different Ollama models (llama2, codellama, etc.)
- **Retrieval**: Tune `k` parameter for more/fewer context chunks

## License

This project is open source and available under the MIT License.