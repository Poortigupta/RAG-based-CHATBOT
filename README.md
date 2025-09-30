# RAG-based Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on PDF documents. The system uses vector embeddings to find relevant document chunks and generates contextual responses using large language models.

## Features

- **Multi-Provider Support**: Supports OpenAI, Google (Gemini), and local (Hugging Face) embedding models
- **PDF Document Processing**: Automatically loads and processes PDF documents
- **Vector Database**: Uses ChromaDB for efficient similarity search
- **Flexible Configuration**: Environment-based configuration for different providers
- **Document Chunking**: Intelligent text splitting with overlap for better context preservation
- **Similarity Search**: Configurable retrieval with relevance scoring

## Architecture

The system consists of three main components:

1. **Document Loader** (`loader.py`): Processes PDF documents and creates vector embeddings
2. **Query Engine** (`query_data.py`): Handles user queries and generates responses
3. **Embedding Comparison** (`compare_embeddings.py`): Utility for comparing embedding vectors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Poortigupta/RAG-based-CHATBOT.git
cd RAG-based-CHATBOT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional PDF processing dependencies:
```bash
pip install "unstructured[pdf]"
```

## Configuration

Create a `.env` file in the project root with your API keys and configuration:

```env
# Choose embedding provider: OPENAI, GOOGLE, or LOCAL
EMBEDDING_PROVIDER=GOOGLE

# Google AI API Configuration (if using GOOGLE provider)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_EMBEDDING_MODEL=text-embedding-004
GOOGLE_CHAT_MODEL=gemini-2.5-flash

# OpenAI API Configuration (if using OPENAI provider)
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small

# Hugging Face Configuration (if using LOCAL provider)
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Query Configuration
RETRIEVAL_K=8
RELEVANCE_THRESHOLD=0.5
```

### Provider Options

- **GOOGLE** (Default): Uses Google's Gemini embeddings and chat models
- **OPENAI**: Uses OpenAI's embedding and chat models
- **LOCAL**: Uses Hugging Face sentence transformers (runs locally, no API key required)

## Usage

### 1. Load Documents

Place your PDF documents in the project directory and run the loader:

```bash
python loader.py
```

This will:
- Load all PDF files from the current directory
- Split documents into chunks
- Generate embeddings
- Store vectors in ChromaDB

### 2. Query the System

Ask questions about your documents:

```bash
python query_data.py "Your question here"
```

Example:
```bash
python query_data.py "What is the main topic of the document?"
```

### 3. Compare Embeddings (Optional)

Test embedding similarity between words:

```bash
python compare_embeddings.py
```

## Project Structure

```
RAG-based-CHATBOT/
├── loader.py              # Document loading and vector database creation
├── query_data.py          # Query processing and response generation
├── compare_embeddings.py  # Embedding comparison utilities
├── requirements.txt       # Python dependencies
├── Samadhan.pdf          # Sample PDF document
├── chroma/               # ChromaDB vector database
│   ├── chroma.sqlite3
│   └── 6b95bac7-acf7-4930-916f-e985ea6b1e4c/
└── .env                  # Environment configuration (create this file)
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Embedding provider (OPENAI/GOOGLE/LOCAL) | GOOGLE |
| `RETRIEVAL_K` | Number of documents to retrieve | 8 |
| `RELEVANCE_THRESHOLD` | Minimum relevance score | 0.5 |
| `GOOGLE_EMBEDDING_MODEL` | Google embedding model | text-embedding-004 |
| `GOOGLE_CHAT_MODEL` | Google chat model | gemini-2.5-flash |
| `EMBEDDING_MODEL` | OpenAI embedding model | text-embedding-3-small |
| `HF_EMBEDDING_MODEL` | Hugging Face model | sentence-transformers/all-MiniLM-L6-v2 |

## Dependencies

Key dependencies include:

- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **OpenAI**: OpenAI API integration
- **Google Generative AI**: Google's Gemini models
- **Sentence Transformers**: Local embedding models
- **PyPDF**: PDF document processing
- **Unstructured**: Document loading utilities

## Troubleshooting

### No Results from Vector Store
If you get "No results returned from vector store", ensure you've run `loader.py` after adding your PDF documents.

### API Key Issues
- For Google: Set `GOOGLE_API_KEY` in your `.env` file
- For OpenAI: Set `OPENAI_API_KEY` in your `.env` file
- For local models: No API key required, but initial model download may take time

### Rate Limiting
If you encounter rate limits, the system will automatically suggest switching to `EMBEDDING_PROVIDER=LOCAL` for offline processing.



## License

This project is open source. Please check the repository for license details.

## Support

For issues and questions, please open an issue on the GitHub repository.