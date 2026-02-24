# app.py - Streamlit interface for UDST Academic Assistant
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from src.retrieval import load_embeddings, retrieve
from src.generation import generate_answer

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
embeddings_dir = "catalog_embeddings"
TOP_K = 6

# Load environment
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

# Set page config
st.set_page_config(
    page_title="UDST Academic Assistant",
    page_icon="📚",
    layout="centered"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load embeddings and model once
@st.cache_resource
def load_rag_system():
    """Load embedding model, embeddings, and chunks"""
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings, chunks = load_embeddings(embeddings_dir)
    return model, embeddings, chunks

try:
    model, embeddings, chunks = load_rag_system()
    chunks_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load RAG system: {e}")
    st.info("Please run `python main.py` first to generate embeddings.")
    st.stop()

# Custom CSS styling
st.markdown("""
<style>
    div.stButton > button {
        width: 100% !important;
        height: 50px !important;
        white-space: normal !important;
    }

    .global-title {
        color: #2B3A8C;
        font-size: 2.5em !important;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #2B3A8C;
        margin-bottom: 25px;
    }

    .chat-history {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **UDST Academic Assistant**
    
    Answers are sourced directly from the UDST Academic Catalog (2022–2023).
    The assistant retrieves relevant catalog excerpts and returns concise, citation-backed answers.
    
    **How it works:**
    1. Semantic search finds the most relevant catalog excerpts
    2. The assistant generates an answer using only those excerpts
    3. Page numbers are provided for traceability
    """)
    st.markdown("---")
    st.success(f"✅ {len(chunks)} chunks loaded - Total of 441 pages")
    st.caption("UDST Academic Assistant Chatbot")

# Main interface
st.markdown('<h1 class="global-title">📚 UDST ACADEMIC ASSISTANT</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 18px; color: #555;">
  Ask questions about UDST academic policies and get instant answers.
</p>
""", unsafe_allow_html=True)

# Chat history container
with st.container():
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# --- SAMPLE QUESTIONS SECTION ---
if not st.session_state.messages:
    st.subheader("💬 Sample Questions:")
    cols = st.columns(2)
    sample_questions = [
        "What is the maximum number of allowed absences before failing?",
        "How do I add/drop or withdraw from a course and what are the deadlines?",
        "What are the graduation requirements for a bachelor's degree?",
        "How is GPA calculated and what are academic standing rules?",
        "How are transfer credits evaluated and applied?",
        "What scholarships or financial aid options are available?"
    ]

    for i, question in enumerate(sample_questions):
        if cols[i % 2].button(question, use_container_width=True, key=f"sample_{i}"):
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.spinner("🔍 Searching academic catalog..."):
                top_chunks = retrieve(question, model, embeddings, chunks, top_k=TOP_K)
                response_text = generate_answer(question, client, top_chunks)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })
            st.rerun()

# --- INPUT SECTION ---
input_container = st.container()
with input_container:
    cols = st.columns([5, 1])
    with cols[0]:
        prompt = st.chat_input("Ask a question about UDST academic policies:")
    with cols[1]:
        if st.button(
            "Clear",
            key="clear_chat",
            use_container_width=True,
            help="Clear conversation history"
        ):
            st.session_state.messages = []
            st.rerun()

# --- PROCESS USER QUERY ---
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.spinner("🔍 Searching and generating answer..."):
        top_chunks = retrieve(prompt, model, embeddings, chunks, top_k=TOP_K)
        response_text = generate_answer(prompt, client, top_chunks)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey;">
    <p>📋 Responses are based on the UDST Academic Catalog</p>
</div>
""", unsafe_allow_html=True)
