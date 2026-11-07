import os
import streamlit as st
from dotenv import load_dotenv

from core.document_loader import DocumentLoader
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline

# Load local .env if present
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI // Neural Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cyberpunk / Dark Mode Modern Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e6edf3;
    }
    .main-header {
        font-family: 'Courier New', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        color: #00F0FF;
        text-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.0rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #161b22;
        border: 1px solid #00F0FF44;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #52f2b1;
        margin-right: 8px;
    }
    .citation-box {
        background-color: #161b22;
        border-left: 3px solid #00F0FF;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "loader" not in st.session_state:
    st.session_state.loader = DocumentLoader(chunk_size=750, chunk_overlap=120)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚡ Neural Control Unit")
    
    # API Key Configuration
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Get a free API key from Google AI Studio (aistudio.google.com)"
    )

    model_choice = st.selectbox(
        "Inference Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📂 Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Index Uploads", use_container_width=True):
            if uploaded_files:
                with st.spinner("Parsing & embedding vectors..."):
                    new_chunks = []
                    for f in uploaded_files:
                        if f.name not in st.session_state.indexed_docs:
                            if f.name.endswith(".pdf"):
                                chunks = st.session_state.loader.load_pdf(f, doc_name=f.name)
                            else:
                                text = f.read().decode("utf-8", errors="ignore")
                                chunks = st.session_state.loader.load_text(text, doc_name=f.name)
                            new_chunks.extend(chunks)
                            st.session_state.indexed_docs.append(f.name)

                    if new_chunks:
                        if api_key:
                            st.session_state.vector_store.set_api_key(api_key)
                        st.session_state.vector_store.add_documents(new_chunks)
                        st.success(f"Indexed {len(new_chunks)} chunks!")
            else:
                st.warning("Please select a file first.")

    with col2:
        if st.button("📑 Load Sample Paper", use_container_width=True):
            sample_path = os.path.join("sample_docs", "transformer_architecture.txt")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = st.session_state.loader.load_text(content, doc_name="Attention_Paper_Summary.txt")
                if api_key:
                    st.session_state.vector_store.set_api_key(api_key)
                st.session_state.vector_store.add_documents(chunks)
                st.session_state.indexed_docs.append("Attention_Paper_Summary.txt")
                st.success("Loaded Transformer Architecture sample!")

    st.markdown("---")
    st.markdown("### 📊 Vector Index Telemetry")
    total_chunks = len(st.session_state.vector_store.chunks)
    st.markdown(f"**Chunks Indexed:** `{total_chunks}`")
    st.markdown(f"**Active Documents:** `{len(set(st.session_state.indexed_docs))}`")
    if st.session_state.indexed_docs:
        for doc in set(st.session_state.indexed_docs):
            st.caption(f"• {doc}")

    if st.button("🗑️ Reset Vector Store", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.indexed_docs = []
        st.session_state.chat_history = []
        st.rerun()

# Instantiate RAG Pipeline
rag_pipeline = RAGPipeline(
    vector_store=st.session_state.vector_store,
    api_key=api_key,
    model_name=model_choice
)

# Main Dashboard View
st.markdown("<div class='main-header'>DOCUMIND // RAG ARCHITECTURE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>High-Dimensional Retrieval Augmented Generation Engine & Research Analyst</div>", unsafe_allow_html=True)

# Render Chat History
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["user"])

    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])

        # Telemetry & Citation Drawer
        if "latency_ms" in turn and turn["latency_ms"] > 0:
            st.markdown(f"<span class='metric-badge'>⚡ {turn['latency_ms']} ms</span><span class='metric-badge'>🤖 {turn.get('model', model_choice)}</span>", unsafe_allow_html=True)

        if "sources" in turn and turn["sources"]:
            with st.expander("📖 Retrieved Evidence & Citations"):
                for src in turn["sources"]:
                    st.markdown(f"""
                    <div class='citation-box'>
                        <b>📄 Document:</b> {src['doc_name']} &nbsp;|&nbsp; <b>Page:</b> {src['page_number']} &nbsp;|&nbsp; <b>Relevance:</b> {src['relevance_score']}
                        <br><br>
                        <i>"{src['snippet']}"</i>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input Field
if query := st.chat_input("Ask a question about your uploaded research papers..."):
    # Append user prompt
    with st.chat_message("user"):
        st.markdown(query)

    # Execute RAG
    with st.chat_message("assistant"):
        with st.spinner("Scanning vector indices & synthesizing response..."):
            result = rag_pipeline.answer_query(query, top_k=4)
            st.markdown(result["answer"])

            if result["latency_ms"] > 0:
                st.markdown(f"<span class='metric-badge'>⚡ {result['latency_ms']} ms</span><span class='metric-badge'>🤖 {result['model']}</span>", unsafe_allow_html=True)

            if result["sources"]:
                with st.expander("📖 Retrieved Evidence & Citations"):
                    for src in result["sources"]:
                        st.markdown(f"""
                        <div class='citation-box'>
                            <b>📄 Document:</b> {src['doc_name']} &nbsp;|&nbsp; <b>Page:</b> {src['page_number']} &nbsp;|&nbsp; <b>Relevance:</b> {src['relevance_score']}
                            <br><br>
                            <i>"{src['snippet']}"</i>
                        </div>
                        """, unsafe_allow_html=True)

    # Save to history
    st.session_state.chat_history.append({
        "user": query,
        "assistant": result["answer"],
        "sources": result["sources"],
        "latency_ms": result["latency_ms"],
        "model": result["model"]
    })
