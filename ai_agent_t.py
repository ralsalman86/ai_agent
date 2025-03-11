import streamlit as st
import requests
import re
import json
from pydantic import BaseModel
from firecrawl import FirecrawlApp
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Firecrawl API function
def scrape_website(url, api_key):
    app = FirecrawlApp(api_key=api_key)
    scrape_result = app.scrape_url(url, params={'formats': ['markdown', 'html']})

    if scrape_result is None:
        st.error(f"Failed to scrape website")
        return None
    else:
        return extract_text_from_html(scrape_result['html'])


# Function to extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=' ')


# Function to clean URLs from markdown content
def clean_urls(content):
    return re.sub(r'http\S+', '', content)


# Function to chunk text
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# Function to retrieve relevant context using BM25
def bm25_retrieve(query, content_chunks):
    tokenized_chunks = [chunk.split() for chunk in content_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    top_chunks = bm25.get_top_n(query.split(), content_chunks, n=3)
    return " ".join(top_chunks)


# Function to retrieve relevant context using embeddings
def embedding_retrieve(query, content_chunks, embed_model):
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embed_model.encode(content_chunks, convert_to_tensor=True)
    dense_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = dense_scores.argsort(descending=True)[:3]
    return " ".join([content_chunks[i] for i in top_indices])


# Function to retrieve relevant context using Hybrid Search
def hybrid_retrieve(query, content_chunks, embed_model):
    bm25_context = bm25_retrieve(query, content_chunks)
    embedding_context = embedding_retrieve(query, content_chunks, embed_model)
    return bm25_context + " " + embedding_context


# Function to retrieve relevant context using FAISS VectorDB
def faiss_retrieve(query, content_chunks, embed_model):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(content_chunks, embeddings)
    docs = vectordb.similarity_search(query, k=3)
    return " ".join([doc.page_content for doc in docs])


# Function to interact with Ollama via HTTP API
def chat_with_model(prompt, context, model='phi4:latest'):
    ollama_url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": model,
        "prompt": f"Context: {context}\n\nUser: {prompt}\n\nAssistant:",
        "stream": False
    }

    response = requests.post(ollama_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get('response', 'No response from model.')
    else:
        st.error(f"Ollama API error: {response.text}")
        return "Error in fetching response."


# Streamlit App
def main():
    st.title('Website Content Chatbot')

    # Initialize session state
    if "scraped_content" not in st.session_state:
        st.session_state.scraped_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Dropdown for selecting the Ollama model
    model_choice = st.selectbox("Select Ollama Model:",
                                ["phi4:latest", "deepseek-coder-v2:latest", "qwen2.5-coder:latest"])

    # Dropdown for selecting RAG method
    rag_method = st.selectbox("Select RAG Method:", ["None", "BM25", "Embeddings", "Hybrid", "FAISS"])

    # Dropdown for selecting embedding model
    embedding_model_name = st.selectbox("Select Embedding Model:", [
        "all-MiniLM-L6-v2", "BAAI/bge-large-en", "intfloat/e5-large-v2"
    ])

    # Load selected embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    api_key = 'fc-f5c2f3300b074f1db5e8ca285e73618f'  # Replace with secure input if needed
    url = st.text_input('Enter the URL to scrape:')

    if st.button('Scrape and Chat'):
        if not api_key or not url:
            st.error('Please provide both the API key and URL.')
        else:
            with st.spinner('Scraping website...'):
                content = scrape_website(url, api_key)

            if content:
                st.session_state.scraped_content = clean_urls(content)
                st.session_state.chat_history = []  # Reset chat history
                st.success('Website content scraped successfully!')

    if st.session_state.scraped_content:
        st.write('### Scraped Content')
        st.text_area('Scraped Content:', st.session_state.scraped_content, height=200, disabled=True)

        st.write('---')
        st.write('## Chat with the Model')

        # Display chat history
        for msg in st.session_state.chat_history:
            role, text = msg
            if role == "user":
                st.chat_message("user").write(text)
            else:
                st.chat_message("assistant").write(text)

        # Chat input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))

            with st.spinner('Generating response...'):
                context = st.session_state.scraped_content
                if rag_method == "BM25":
                    content_chunks = chunk_text(context)
                    context = bm25_retrieve(user_input, content_chunks)
                elif rag_method == "Embeddings":
                    content_chunks = chunk_text(context)
                    context = embedding_retrieve(user_input, content_chunks, embedding_model)
                elif rag_method == "Hybrid":
                    content_chunks = chunk_text(context)
                    context = hybrid_retrieve(user_input, content_chunks, embedding_model)
                elif rag_method == "FAISS":
                    content_chunks = chunk_text(context)
                    context = faiss_retrieve(user_input, content_chunks, embedding_model)

                response = chat_with_model(user_input, context, model_choice)
                st.session_state.chat_history.append(("assistant", response))

            st.rerun()


if __name__ == '__main__':
    main()
