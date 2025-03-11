# Website Content Chatbot with RAG and Ollama Models

This Streamlit application scrapes website content, processes it using various Retrieval-Augmented Generation (RAG) methods, and allows interaction with different Ollama language models.

## Features
- **Scrape website content** using Firecrawl API.
- **Clean and process text** extracted from HTML.
- **Support for multiple RAG methods:**
  - **BM25** (Sparse Retrieval)
  - **Embeddings-based Retrieval** (Dense Retrieval)
  - **Hybrid Retrieval** (BM25 + Embeddings)
  - **FAISS VectorDB Retrieval** (Enterprise-grade RAG)
- **Select from multiple Ollama models** to generate responses.
- **Choose embedding models** for similarity search.

## Installation
To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the Streamlit app:
   ```bash
   streamlit run ai_agent_t.py
   ```
2. Enter the URL of a website to scrape.
3. Select the desired **Ollama model**, **RAG method**, and **embedding model**.
4. Ask questions based on the scraped content.

## Configuration
- The Ollama API should be running locally at `http://localhost:11434`.
- Update `api_key` with a valid Firecrawl API key.

## Dependencies
- **Streamlit**: Web framework for interactive UI
- **Requests**: API communication
- **Firecrawl**: Web scraping API
- **Sentence Transformers**: Embeddings for RAG
- **BM25 (rank-bm25)**: Sparse retrieval
- **LangChain**: Text splitting and vector search
- **BeautifulSoup (bs4)**: HTML parsing
- **FAISS**: Vector database for efficient similarity search

## License
This project is open-source under the MIT License.
