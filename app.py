import streamlit as st
import numpy as np
from pypdf import PdfReader
import tiktoken
import faiss
import requests
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
import time

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Free Models
FREE_MODELS = {
    "Llama 3.2 3B": "meta-llama/llama-3.2-3b-instruct:free",
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",
    "Mistral 7B": "mistralai/mistral-7b-instruct:free",
    "Qwen 2.5 7B": "qwen/qwen-2.5-7b-instruct:free",
}

# Custom OpenRouter Chat Function with retry logic
def chat_with_openrouter(prompt, model, temperature=0.2, max_retries=3):
    """Send a request to OpenRouter API with retry logic"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return "⚠️ Rate limit exceeded. Please wait a moment and try again, or switch to a different model in the sidebar."
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            return "⚠️ Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return f"⚠️ API Error: {str(e)}"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"
    
    return "⚠️ Failed after multiple retries. Please try again later."

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def read_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"[ERROR] while reading PDF: {str(e)}")
        return ""

def chunk_text(text):
    max_tokens = 300  # Increased chunk size for better context
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
    vector = embedding_model.embed_query(text)
    return vector

def build_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dims = len(embeddings[0])
    index = faiss.IndexFlatL2(dims)
    index.add(np.array(embeddings))
    return index

def handle_query(query, index, chunks, model):
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
    
    response = chat_with_openrouter(prompt, model)
    return response

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "google/gemini-2.0-flash-exp:free"

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
    st.markdown("<h2 style='color: orangered;'>💬 Personalized Chatbot</h2>", unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 Select Model")
        selected_model_name = st.selectbox(
            "Choose a free model:",
            options=list(FREE_MODELS.keys()),
            index=0
        )
        st.session_state.selected_model = FREE_MODELS[selected_model_name]
        
        st.caption(f"Current: {selected_model_name}")
        
        # Tips
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- If you get rate limited, try switching models")
        st.markdown("- Wait 1-2 minutes between requests")
        st.markdown("- Free tier has usage limits")
        
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()
        
        if st.button("Reset Document"):
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
    
    # Display chat UI
    chatui()
    
    # Chat input
    if st.session_state.faiss_index:
        question = st.chat_input("Ask a question about the document:")
        if question:
            with st.spinner("Generating answer..."):
                answer = handle_query(
                    question, 
                    st.session_state.faiss_index, 
                    st.session_state.chunks,
                    st.session_state.selected_model
                )
                
                st.session_state.history.append({"role": "user", "content": question})
                st.session_state.history.append({"role": "bot", "content": answer})
                
                st.rerun()
    else:
        st.info("👆 Please upload a PDF file to start chatting!")

if __name__ == "__main__":
    main()