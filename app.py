import streamlit as st
import numpy as np
from pypdf import PdfReader
import tiktoken
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


API_KEY = "AIzaSyDbvETkEt3FC4x-Ir3UodhvtQMRDQbEDqw"


chat_model = ChatGoogleGenerativeAI(api_key=API_KEY, model='gemini-1.5-flash', temparature=0.2)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, model="models/embedding-001")

def read_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"[ERROR] while reading PDF: {str(e)}")

def chunk_text(text):
    max_tokens = 30
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

def handle_query(query, index, chunks):
    query_emb = np.array(get_embedding(query)).reshape(1, -1)
    distances, indices = index.search(query_emb, k=4)
    relavent_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relavent_chunks)
    
    prompt = PromptTemplate(
        template="""
        You are an helpful assistant to assist healthcare professionals.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:""",
        input_variables=['context', 'query']
    )
    
    parser = StrOutputParser()
    chain = prompt | chat_model | parser
    response = chain.invoke({"query": query, 'context': context})
    return response

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []
st.set_page_config(page_title="Chatbot", layout="wide")

def chatui():

    st.markdown(
        """
        <style>
        .chat-box {
            max-width: 700px;
            float: right;
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


def Ui():
    st.markdown("<h2 style='color: orangered;'>💬 Chat Assistant</h2>", unsafe_allow_html=True)
    
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    chunks = []
    faiss_index = None
    
    if pdf_file:
        text = read_pdf(pdf_file)
        chunks = chunk_text(text)
        faiss_index = build_index(chunks)

        st.success(f"Document loaded and processed successfully !!")
    
    if faiss_index:
        question = st.chat_input("Ask a question about the document:")
        if question:
            answer = handle_query(question, faiss_index, chunks)
            
            st.session_state.history.append({"role": "user", "content": question})

            st.session_state.history.append({"role": "bot", "content": answer})

            chatui()

Ui()