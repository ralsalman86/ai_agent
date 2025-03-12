import streamlit as st
import requests
import re
import json
import os
from groq import Groq
from pydantic import BaseModel
from firecrawl import FirecrawlApp
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Hardcoded API Keys
groq_api_key = "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"
firecrawl_api_key = "fc-f5c2f3300b074f1db5e8ca285e73618f"


# Function to fetch available Groq models
def fetch_groq_models():
    return ['deepseek-r1-distill-qwen-32b', 'llama-3.1-8b-instant',  'deepseek-r1-distill-llama-70b',
            'distil-whisper-large-v3-en', '']


# Firecrawl API function
def scrape_website(url):
    app = FirecrawlApp(api_key=firecrawl_api_key)
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


# Function to retrieve relevant context using FAISS
def faiss_retrieve(query, content_chunks, embed_model):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(content_chunks, embeddings)
    docs = vectordb.similarity_search(query, k=3)
    return " ".join([doc.page_content for doc in docs])


# Function to interact with Groq API
def chat_with_groq(prompt, context, model='llama3-8b'):
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": context}
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content


# Streamlit App
def main():
    st.title('Website Content Chatbot with Groq')

    # Initialize session state
    if "scraped_content" not in st.session_state:
        st.session_state.scraped_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    available_models = fetch_groq_models()
    model_choice = st.selectbox("Select Groq Model:", available_models if available_models else ["llama3-8b"])

    # RAG method selection
    rag_method = st.selectbox("Select RAG Method:", ["BM25", "None",  "Embeddings", "FAISS"])
    embedding_model_name = st.selectbox("Select Embedding Model:", ["all-MiniLM-L6-v2", "BAAI/bge-large-en"])
    embedding_model = SentenceTransformer(embedding_model_name)

    url = st.text_input('Enter the URL to scrape:')

    if st.button('Scrape and Chat'):
        if not url:
            st.error('Please provide a URL.')
        else:
            with st.spinner('Scraping website...'):
                content = scrape_website(url)

            if content:
                st.session_state.scraped_content = clean_urls(content)
                st.session_state.chat_history = []
                st.success('Website content scraped successfully!')

    if st.session_state.scraped_content:
        st.write('### Scraped Content')
        st.text_area('Scraped Content:', st.session_state.scraped_content, height=200, disabled=True)
        st.write('---')
        st.write('## Chat with the Model')

        # Display chat history
        for role, text in st.session_state.chat_history:
            st.chat_message(role).write(text)

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
                elif rag_method == "FAISS":
                    content_chunks = chunk_text(context)
                    context = faiss_retrieve(user_input, content_chunks, embedding_model)
                response = chat_with_groq(user_input, context, model_choice)
                st.session_state.chat_history.append(("assistant", response))
            st.rerun()


if __name__ == '__main__':
    main()
