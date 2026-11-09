import os
import streamlit as st
from core.ui_shared import init_app, render_sidebar

st.set_page_config(
    page_title="DocuMind — Knowledge Analytics & Control Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

pipeline = init_app()
render_sidebar()

# Hero Header Showcase
st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">✨ Executive Knowledge Analytics & Document Intelligence</div>
    <div class="brand-heading">DocuMind</div>
    <div class="brand-tagline">
        Autonomous document ingestion, verbatim-grounded Q&A, cognitive mind mapping, gamified testing, and academic exam synthesis.
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.status_banner:
    b_type, b_msg = st.session_state.status_banner
    if b_type == "success":
        st.success(b_msg)
    elif b_type == "warning":
        st.warning(b_msg)
    elif b_type == "info":
        st.info(b_msg)

active_docs = list(set(st.session_state.indexed_docs))
total_chunks = len(st.session_state.vector_store.chunks)
total_words = sum(len(c.text.split()) for c in st.session_state.vector_store.chunks)
est_read_min = max(1, total_words // 200) if total_words > 0 else 0
study_turns = len(st.session_state.chat_history)
has_quiz = 1 if st.session_state.get("interactive_quiz") else 0

# 1. Executive KPI Telemetry Ribbon
st.markdown(f"""
<div class="dashboard-kpi-grid">
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-icon">📚</span>
            <span class="kpi-badge">FILES</span>
        </div>
        <div class="kpi-val">{len(active_docs)}</div>
        <div class="kpi-lbl">Active Documents</div>
        <div class="kpi-sub">{"Indexed & in memory" if active_docs else "No documents loaded"}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-icon">🧩</span>
            <span class="kpi-badge">PASSAGES</span>
        </div>
        <div class="kpi-val">{total_chunks}</div>
        <div class="kpi-lbl">Searchable Chunks</div>
        <div class="kpi-sub">{f"~{total_words:,} total words" if total_words > 0 else "Ready for ingestion"}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-icon">⏱️</span>
            <span class="kpi-badge">READ TIME</span>
        </div>
        <div class="kpi-val">{est_read_min}m</div>
        <div class="kpi-lbl">Est. Reading Time</div>
        <div class="kpi-sub">Calculated at 200 wpm</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-icon">🎯</span>
            <span class="kpi-badge">ENGAGEMENT</span>
        </div>
        <div class="kpi-val">{study_turns}</div>
        <div class="kpi-lbl">Questions Explored</div>
        <div class="kpi-sub">{"Active research session" if study_turns > 0 else "Session ready"}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-icon">⚡</span>
            <span class="kpi-badge">ENGINE</span>
        </div>
        <div class="kpi-val">100%</div>
        <div class="kpi-lbl">Verbatim Grounding</div>
        <div class="kpi-sub">Sublinear TF-IDF + Gemini</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Knowledge Base Overview or Quick-Start Onboarding
if active_docs:
    st.markdown("""
    <div class="dash-section">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">📂 Loaded Knowledge Inventory</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Detailed breakdown of active textbooks, research papers, and notes.</div>
            </div>
            <div class="pill-tag">🟢 Live In-Memory</div>
        </div>
    """, unsafe_allow_html=True)

    for doc in active_docs:
        doc_chunks = [c for c in st.session_state.vector_store.chunks if c.doc_name == doc]
        doc_words = sum(len(c.text.split()) for c in doc_chunks)
        ext = doc.split(".")[-1].upper() if "." in doc else "DOC"
        st.markdown(f"""
        <div class="inventory-item">
            <div class="inventory-title">
                <span>📖</span>
                <span><b>{doc}</b></span>
                <span class="kpi-badge" style="font-size: 0.65rem;">{ext}</span>
            </div>
            <div class="inventory-meta">
                <span>🧩 <b>{len(doc_chunks)}</b> passages</span>
                <span>📝 ~<b>{doc_words:,}</b> words</span>
                <span style="color: #34d399;">● Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Instant Knowledge Search Sandbox directly on Dashboard
    st.markdown("""
    <div class="dash-section">
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
            ⚡ Instant Knowledge Search Sandbox
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px;">
            Preview sub-second passage retrieval directly from your book index before launching deep Q&A.
        </div>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input(
        "Type any keyword or concept to test retrieval:",
        placeholder="e.g. self-attention, multithreading, virtual memory...",
        key="dash_search_box"
    )

    if search_query:
        results = st.session_state.vector_store.search(search_query, top_k=2)
        if results:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #a5b4fc; margin-bottom: 8px;'>🔍 Top Retrieved Passages:</div>", unsafe_allow_html=True)
            for res in results:
                st.markdown(f"""
                <div class="citation-container" style="margin-bottom: 10px;">
                    <div class="citation-title">📄 {res.doc_name} • Page {res.page_number} (Similarity Score: {res.relevance_score})</div>
                    <div style="font-size: 0.88rem; color: #e2e8f0; line-height: 1.5;">"{res.snippet}"</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("💬 Ask Full Question in Q&A Reader", key="btn_jump_qa"):
                st.session_state.suggested_prompt = search_query
                st.switch_page("pages/1_💬_QA_Reader.py")
        else:
            st.info("No matching passages found for this query.")
else:
    # Onboarding Empty State Card
    st.markdown("""
    <div class="dash-section" style="text-align: center; padding: 36px 24px;">
        <div style="font-size: 2.2rem; margin-bottom: 10px;">📚</div>
        <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff; margin-bottom: 6px;">
            Welcome to DocuMind Knowledge Hub
        </div>
        <div style="font-size: 0.90rem; color: #94a3b8; max-width: 650px; margin: 0 auto 20px auto; line-height: 1.6;">
            Get started by uploading your textbook or notes in the left sidebar, or click below to immediately load our pre-configured sample research paper on <b>Transformer Architecture</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    ob_col1, ob_col2, ob_col3 = st.columns([1, 2, 1])
    with ob_col2:
        if st.button("📑 Load Sample Transformer Research Paper", use_container_width=True, key="dash_load_sample"):
            sample_path = os.path.join("sample_docs", "transformer_architecture.txt")
            if os.path.exists(sample_path):
                with st.spinner("Indexing sample paper..."):
                    with open(sample_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    chunks = st.session_state.loader.load_text(content, doc_name="Transformer_Architecture_Paper.txt")
                    st.session_state.vector_store.set_api_key(st.session_state.api_key)
                    st.session_state.vector_store.add_documents(chunks)
                    if "Transformer_Architecture_Paper.txt" not in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.append("Transformer_Architecture_Paper.txt")
                    st.session_state.status_banner = ("success", f"Loaded sample document ({len(chunks)} passages indexed).")
                    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# 4. Dedicated Workspaces Command Center
st.markdown("### 🚀 Dedicated Study & Research Workspaces")
st.caption("Select an intelligence workspace from below or use the sidebar navigation:")

col_ws1, col_ws2 = st.columns(2)

with col_ws1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-bottom: 4px;">
            💬 Strict Q&A Reader & Split Viewer
        </div>
        <div style="color: #94a3b8; font-size: 0.90rem; margin-bottom: 12px; line-height: 1.5;">
            Typewriter streaming, synchronized split-page book browsing, in-browser speech playback, and 4 teaching personas (Professor, Feynman, Cram Coach, Analyst).
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Launch Q&A Reader", use_container_width=True, key="dash_btn_qa"):
        st.switch_page("pages/1_💬_QA_Reader.py")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-bottom: 4px;">
            📝 Exam Paper & Deep Research Lab
        </div>
        <div style="color: #94a3b8; font-size: 0.90rem; margin-bottom: 12px; line-height: 1.5;">
            Formulate university question papers (Section A, B, C) with confidential marking rubrics & multi-hop research reports.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Launch Exam & Research", use_container_width=True, key="dash_btn_exam"):
        st.switch_page("pages/3_📝_Exam_&_Research.py")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-bottom: 4px;">
            📥 Portable Export Suite & Anki
        </div>
        <div style="color: #94a3b8; font-size: 0.90rem; margin-bottom: 12px; line-height: 1.5;">
            Download Anki spaced repetition flashcards (.csv), standalone HTML reports, and plain text transcripts.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Launch Export Suite", use_container_width=True, key="dash_btn_export"):
        st.switch_page("pages/5_📥_Export_&_Anki.py")

with col_ws2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-bottom: 4px;">
            🎮 Gamified Live Quiz & Scorecard
        </div>
        <div style="color: #94a3b8; font-size: 0.90rem; margin-bottom: 12px; line-height: 1.5;">
            Interactive live test with radio button choices, instant scoring scorecard, book explanations, and balloon celebrations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Launch Live Quiz", use_container_width=True, key="dash_btn_quiz"):
        st.switch_page("pages/2_🎮_Live_Quiz.py")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-bottom: 4px;">
            🧠 Cognitive Mind Map & Summaries
        </div>
        <div style="color: #94a3b8; font-size: 0.90rem; margin-bottom: 12px; line-height: 1.5;">
            Visual Mermaid.js concept graphs, 1-click chapter cheat sheets, and side-by-side multi-document comparison matrices.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Launch Mind Map & Summaries", use_container_width=True, key="dash_btn_mm"):
        st.switch_page("pages/4_🧠_Mind_Map_&_Summary.py")

st.markdown("<br>", unsafe_allow_html=True)

# 5. System Health & Infrastructure Diagnostics
with st.expander("⚡ System Health & Neural Engine Telemetry", expanded=False):
    diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(4)
    with diag_c1:
        st.markdown("**LLM Engine:**")
        st.code(f"{pipeline.provider.upper()} ({pipeline.model_name})", language="text")
    with diag_c2:
        st.markdown("**Response Streaming:**")
        st.code("Typewriter Generator Active", language="text")
    with diag_c3:
        st.markdown("**Voice Synthesis:**")
        st.code("Web Speech Synthesis API", language="text")
    with diag_c4:
        st.markdown("**Vector Store:**")
        st.code("Sublinear TF-IDF (Offline)", language="text")
