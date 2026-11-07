import streamlit as st
from core.ui_shared import init_app, render_sidebar

st.set_page_config(
    page_title="Q&A Reader — DocuMind",
    page_icon="💬",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">📖 Verbatim Document Research</div>
    <div class="brand-heading">Strict Q&A Reader</div>
    <div class="brand-tagline">
        Ask any question. Answers are strictly extracted and verbatim-quoted from your book's exact sentences with page citations.
    </div>
</div>
""", unsafe_allow_html=True)

active_docs = list(set(st.session_state.indexed_docs))
total_chunks = len(st.session_state.vector_store.chunks)

if active_docs:
    st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;'>💡 <b>Quick-test starter questions:</b></div>", unsafe_allow_html=True)
    chip1, chip2, chip3 = st.columns(3)
    with chip1:
        if st.button("👉 What is multithreading?", use_container_width=True):
            st.session_state.suggested_prompt = "What is multithreading?"
    with chip2:
        if st.button("👉 What is self-attention?", use_container_width=True):
            st.session_state.suggested_prompt = "What is self-attention and how does it work?"
    with chip3:
        if st.button("👉 Summarize key concepts", use_container_width=True):
            st.session_state.suggested_prompt = "Summarize the core concepts covered in the loaded document."
else:
    st.info("💡 **Getting Started:** Upload your book or folder PDFs in the left sidebar, or click **'📑 Load Sample'** to start.")

# Render Conversation
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])
        if "sources" in turn and turn["sources"]:
            with st.expander("📖 Exact Source Excerpts & Page Proof"):
                for src in turn["sources"]:
                    st.markdown(f"""
                    <div class="citation-container">
                        <div class="citation-title">📄 {src['doc_name']} • Page {src['page_number']} (Score: {src['relevance_score']})</div>
                        <div style="font-style: italic; color: #e2e8f0;">"{src['snippet']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

query = st.chat_input("Ask any question from your uploaded book...")
if st.session_state.suggested_prompt:
    query = st.session_state.suggested_prompt
    st.session_state.suggested_prompt = None

if query:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching book pages & quoting exact sentences..."):
            result = pipeline.answer_query(query, top_k=5)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📖 Exact Source Excerpts & Page Proof"):
                    for src in result["sources"]:
                        st.markdown(f"""
                        <div class="citation-container">
                            <div class="citation-title">📄 {src['doc_name']} • Page {src['page_number']} (Score: {src['relevance_score']})</div>
                            <div style="font-style: italic; color: #e2e8f0;">"{src['snippet']}"</div>
                        </div>
                        """, unsafe_allow_html=True)

    st.session_state.chat_history.append({
        "user": query,
        "assistant": result["answer"],
        "sources": result["sources"],
        "latency_ms": result["latency_ms"],
        "model": result["model"]
    })
