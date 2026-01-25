import streamlit as st
import numpy as np
from pypdf import PdfReader
import tiktoken
import faiss
import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

def chat_with_gemini(prompt, temperature=0.2):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Replace your existing HuggingFaceEmbeddings with this:
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

def read_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return ""

def chunk_text(text):
    max_tokens = 300
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks, chunk, tokens = [], [], 0
    for word in words:
        token_count = len(tokenizer.encode(word))
        if tokens + token_count > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, tokens = [word], token_count
        else:
            chunk.append(word)
            tokens += token_count
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def get_embedding(text):
    return embedding_model.embed_query(text)

def build_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dims = len(embeddings[0])
    index = faiss.IndexFlatL2(dims)
    index.add(np.array(embeddings))
    return index

def handle_query(query, index, chunks):
    query_emb = np.array(get_embedding(query)).reshape(1, -1)
    distances, indices = index.search(query_emb, k=4)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are a helpful assistant answering questions based on the following context:

Context:
{context}

Question:
{query}

Please provide a clear and concise answer based only on the context provided above."""
    
    return chat_with_gemini(prompt)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.set_page_config(page_title="PDF Chatbot", layout="wide")

def chatui():
    st.markdown(
        """
        <style>
        .chat-box {
            max-width: 700px;
            margin: auto;
            border: 2px solid orangered;
            border-radius: 15px;
            padding: 15px;
            background-color: #f8f9fa;
        }

        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
        }
        .user-bubble {
            background-color: #dcf8c6;
            color: black;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px;
            text-align: right;
            display: inline-block;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #333;
            color: #fff;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px;
            text-align: left;
            display: inline-block;
            max-width: 70%;
            border: 1px solid #ddd;
            float: left;
            clear: both;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.history:
        chat_html = '<div class="chat-box"><div class="chat-container">'
        for msg in st.session_state.history:
            if msg["role"] == "user":
                chat_html += f'<div class="user-bubble">{msg["content"]}</div>'
            else:
                chat_html += f'<div class="bot-bubble">{msg["content"]}</div>'
        chat_html += "</div></div>"
        st.markdown(chat_html, unsafe_allow_html=True)

def main():
    # Chat history
    if st.session_state.history:
        with st.sidebar:
            st.markdown("### 📜 Past Questions")
            for i, msg in enumerate(st.session_state.history):
                if msg["role"] == "user":
                
                    response = "No response available."
                
                    if i + 1 < len(st.session_state.history) and st.session_state.history[i+1]["role"] == "bot":
                        response = st.session_state.history[i+1]["content"]
                    
                    with st.expander(f"{msg['content']}"):
                        st.write(response)

    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("<h2 style='color: orangered;'>💬 PDF Chatbot</h2>", unsafe_allow_html=True)
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Doc", use_container_width=True):
            st.session_state.faiss_index = None
            st.session_state.chunks = []
            st.session_state.history = []
            st.rerun()
    
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if pdf_file:
        if st.session_state.faiss_index is None:
            with st.spinner("Processing PDF..."):
                text = read_pdf(pdf_file)
                if text:
                    st.session_state.chunks = chunk_text(text)
                    st.session_state.faiss_index = build_index(st.session_state.chunks)
                    st.success(f"✅ Document loaded! {len(st.session_state.chunks)} chunks created.")
                else:
                    st.error("Failed to read PDF content.")
    
    chatui()
    
    if st.session_state.faiss_index:
        question = st.chat_input("Ask a question about the document:")
        if question:
            with st.spinner("Generating answer..."):
                answer = handle_query(question, st.session_state.faiss_index, st.session_state.chunks)
                
                st.session_state.history.append({"role": "user", "content": question})
                st.session_state.history.append({"role": "bot", "content": answer})
                
                st.rerun()
    else:
        st.info("👆 Please upload a PDF file to start chatting!")
if __name__ == "__main__":
    main()