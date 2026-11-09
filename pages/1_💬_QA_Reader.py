import time
import streamlit as st
from core.ui_shared import init_app, render_sidebar, render_audio_player

st.set_page_config(
    page_title="Q&A Reader — DocuMind",
    page_icon="💬",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

# Page Session Initializations
if "reader_persona" not in st.session_state:
    st.session_state.reader_persona = "Professor"
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "💬 Full Chat View"
if "selected_page_num" not in st.session_state:
    st.session_state.selected_page_num = 1
if "selected_doc_name" not in st.session_state:
    st.session_state.selected_doc_name = None
if "last_followups" not in st.session_state:
    st.session_state.last_followups = []

# Hero Header Showcase
st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">📖 Verbatim Document Research & Interactive Split-Reader</div>
    <div class="brand-heading">Strict Q&A Reader</div>
    <div class="brand-tagline">
        Interactive research with real-time streaming, multi-persona reasoning (Professor, Feynman, Exam Coach), audio playback, and synchronized split-page browsing.
    </div>
</div>
""", unsafe_allow_html=True)

active_docs = list(set(st.session_state.indexed_docs))
total_chunks = len(st.session_state.vector_store.chunks)

if not st.session_state.selected_doc_name and active_docs:
    st.session_state.selected_doc_name = active_docs[0]

# Control Toolbar: Persona Selection & View Mode
tool_c1, tool_c2 = st.columns([1, 1])

with tool_c1:
    persona_map = {
        "🎓 Academic Professor (Rigorous & Cited)": "Professor",
        "💡 Feynman Explainer (Intuitive & Analogies)": "Feynman",
        "⚡ Exam Cram Coach (High-Yield Bullets & Traps)": "Exam Cram",
        "💼 Executive Analyst (Strategic & Trade-Offs)": "Analyst"
    }
    selected_label = st.selectbox(
        "🧠 AI Teaching Persona:",
        options=list(persona_map.keys()),
        index=0,
        key="qa_persona_select"
    )
    current_persona = persona_map[selected_label]
    st.session_state.reader_persona = current_persona

with tool_c2:
    view_options = ["💬 Full Chat View", "📖 Split-Screen Document Viewer"]
    chosen_view = st.radio(
        "🖥️ Workspace Layout:",
        options=view_options,
        index=0 if st.session_state.view_mode == "💬 Full Chat View" else 1,
        horizontal=True,
        key="qa_layout_toggle"
    )
    st.session_state.view_mode = chosen_view

st.markdown("<hr style='margin: 12px 0 18px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

# Helper function to render a conversation turn
def render_turn(idx: int, turn: dict):
    with st.chat_message("user"):
        st.markdown(turn["user"])

    with st.chat_message("assistant"):
        p_name = turn.get("persona", "Professor")
        p_badge_icon = "🎓" if p_name == "Professor" else ("💡" if p_name == "Feynman" else ("⚡" if p_name == "Exam Cram" else "💼"))
        st.markdown(f'<div class="persona-badge">{p_badge_icon} Mode: {p_name}</div>', unsafe_allow_html=True)
        st.markdown(turn["assistant"])

        # 1-Click Audio Speech Player
        render_audio_player(turn["assistant"], f"audio_{idx}")

        if turn.get("sources"):
            with st.expander("📖 Exact Source Excerpts & Page Proof"):
                for src in turn["sources"]:
                    st.markdown(f"""
                    <div class="citation-container">
                        <div class="citation-title">📄 {src['doc_name']} • Page {src['page_number']} (Score: {src['relevance_score']})</div>
                        <div style="font-style: italic; color: #e2e8f0; font-size: 0.88rem;">"{src['snippet']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                    # In split mode, button to jump viewer to this exact page
                    if st.session_state.view_mode == "📖 Split-Screen Document Viewer":
                        if st.button(f"🔍 Jump Document Viewer to Page {src['page_number']}", key=f"jump_{idx}_{src['page_number']}_{src['relevance_score']}"):
                            st.session_state.selected_page_num = src['page_number']
                            st.session_state.selected_doc_name = src['doc_name']
                            st.rerun()

# Layout Branch: Split-Screen vs Full Chat
if st.session_state.view_mode == "📖 Split-Screen Document Viewer":
    col_doc, col_chat = st.columns([1, 1], gap="medium")

    # LEFT COLUMN: Interactive Document & Page Reader
    with col_doc:
        st.markdown("""
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span>📖 Live Book & Page Browser</span>
        </div>
        """, unsafe_allow_html=True)

        if active_docs:
            sel_doc = st.selectbox(
                "Active Document:",
                options=active_docs,
                index=active_docs.index(st.session_state.selected_doc_name) if st.session_state.selected_doc_name in active_docs else 0,
                key="split_doc_select"
            )
            st.session_state.selected_doc_name = sel_doc

            doc_pages = pipeline.get_document_pages(sel_doc)
            page_numbers = sorted(list(doc_pages.keys())) if doc_pages else [1]
            min_p = min(page_numbers)
            max_p = max(page_numbers)

            # Ensure current selected page is valid for this doc
            if st.session_state.selected_page_num not in page_numbers:
                st.session_state.selected_page_num = min_p

            page_c1, page_c2 = st.columns([3, 1])
            with page_c1:
                cur_page = st.select_slider(
                    "Navigate Page:",
                    options=page_numbers,
                    value=st.session_state.selected_page_num,
                    key="split_page_slider"
                )
                st.session_state.selected_page_num = cur_page
            with page_c2:
                st.markdown(f"<div style='font-size: 0.78rem; color: #94a3b8; margin-top: 26px; text-align: right;'>Total Pages: <b>{len(page_numbers)}</b></div>", unsafe_allow_html=True)

            page_text = doc_pages.get(cur_page, "No text content found on this page.")
            words_on_page = len(page_text.split())

            st.markdown(f"""
            <div class="book-page-viewer">
                <div class="book-page-header">
                    <div>
                        <span style="font-weight: 700; color: #ffffff; font-size: 0.92rem;">📄 {sel_doc}</span>
                        <span class="pill-tag" style="margin-left: 8px; font-size: 0.70rem; padding: 2px 8px;">PAGE {cur_page} OF {max_p}</span>
                    </div>
                    <span style="font-size: 0.75rem; color: #64748b;">~{words_on_page} words</span>
                </div>
                <div class="book-page-body">
{page_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No documents loaded. Upload a book in the sidebar to browse pages.")

    # RIGHT COLUMN: Chat Area
    with col_chat:
        st.markdown(f"""
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
            <span>💬 Interactive Research Conversation</span>
            <span class="pill-tag" style="font-size: 0.70rem;">⚡ Streaming Active</span>
        </div>
        """, unsafe_allow_html=True)

        for i, turn in enumerate(st.session_state.chat_history):
            render_turn(i, turn)

else:
    # Full Chat View Mode
    if active_docs and not st.session_state.chat_history:
        st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;'>💡 <b>Quick-start starter questions:</b></div>", unsafe_allow_html=True)
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

    for i, turn in enumerate(st.session_state.chat_history):
        render_turn(i, turn)

# Dynamic Contextual Follow-up Chips
if st.session_state.last_followups:
    st.markdown("""
    <div style="margin-top: 14px; margin-bottom: 8px; font-size: 0.82rem; font-weight: 700; color: #a5b4fc; display: flex; align-items: center; gap: 6px;">
        <span>⚡ Suggested Follow-up Inquiries:</span>
    </div>
    """, unsafe_allow_html=True)
    f_cols = st.columns(len(st.session_state.last_followups))
    for idx, f_q in enumerate(st.session_state.last_followups):
        with f_cols[idx]:
            if st.button(f"👉 {f_q}", key=f"btn_followup_{idx}", use_container_width=True):
                st.session_state.suggested_prompt = f_q
                st.rerun()

# Chat Input Handler
query = st.chat_input("Ask any question from your uploaded book...")
if st.session_state.suggested_prompt:
    query = st.session_state.suggested_prompt
    st.session_state.suggested_prompt = None

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        p_name = st.session_state.reader_persona
        p_badge_icon = "🎓" if p_name == "Professor" else ("💡" if p_name == "Feynman" else ("⚡" if p_name == "Exam Cram" else "💼"))
        st.markdown(f'<div class="persona-badge">{p_badge_icon} Mode: {p_name}</div>', unsafe_allow_html=True)

        start_time = time.time()
        stream_gen, sources = pipeline.stream_answer_query(
            query=query,
            top_k=5,
            persona=p_name
        )
        # Real-time fluid typewriter streaming output
        full_response = st.write_stream(stream_gen)
        latency_ms = int((time.time() - start_time) * 1000)

        # In-Browser Voice Audio Player
        render_audio_player(full_response, f"audio_new_{len(st.session_state.chat_history)}")

        if sources:
            with st.expander("📖 Exact Source Excerpts & Page Proof"):
                for src in sources:
                    st.markdown(f"""
                    <div class="citation-container">
                        <div class="citation-title">📄 {src['doc_name']} • Page {src['page_number']} (Score: {src['relevance_score']})</div>
                        <div style="font-style: italic; color: #e2e8f0; font-size: 0.88rem;">"{src['snippet']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Generate 3 contextual follow-up exploration questions
    followups = pipeline.generate_followup_questions(query, full_response)
    st.session_state.last_followups = followups

    st.session_state.chat_history.append({
        "user": query,
        "assistant": full_response,
        "persona": p_name,
        "sources": sources,
        "latency_ms": latency_ms,
        "model": pipeline.model_name
    })
    st.rerun()
