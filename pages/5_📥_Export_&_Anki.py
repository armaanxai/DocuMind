import streamlit as st
from core.ui_shared import init_app, render_sidebar

st.set_page_config(
    page_title="Export & Study Notes — DocuMind",
    page_icon="📥",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">📥 Portable Study Package</div>
    <div class="brand-heading">Export Notes & Flashcards</div>
    <div class="brand-tagline">
        Download study notes that open instantly in Windows (Notepad, Chrome, Word, or Excel) or view directly on screen.
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Export Session States
if "anki_data" not in st.session_state:
    st.session_state.anki_data = None

if "exported_notes_text" not in st.session_state:
    st.session_state.exported_notes_text = None

if "exported_notes_html" not in st.session_state:
    st.session_state.exported_notes_html = None

if not st.session_state.vector_store.chunks:
    st.info("💡 **Getting Started:** Upload your book in the sidebar, or click **'📑 Load Sample'** to generate study exports.")
else:
    exp_col1, exp_col2 = st.columns(2)

    # ----------------- SECTION 1: ANKI FLASHCARDS -----------------
    with exp_col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-weight: 700; font-size: 1.1rem; color: #a5b4fc; margin-bottom: 2px;">
                🗂️ Anki Spaced Repetition Flashcards
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 12px;">
                Generate active recall flashcards from your book. Opens directly in Microsoft Excel or Anki.
            </div>
        </div>
        """, unsafe_allow_html=True)

        anki_topic = st.text_input("Flashcard Topic or Chapter", placeholder="Leave blank for entire book", key="page_anki_topic")
        
        if st.button("⚡ Generate Flashcard Deck", use_container_width=True, key="btn_gen_anki"):
            with st.spinner("Extracting key facts and creating flashcards..."):
                csv_result = pipeline.generate_anki_csv(topic=anki_topic, count=10)
                st.session_state.anki_data = csv_result
                st.rerun()

        # Persistent Download & Preview Area (Will NEVER disappear on click!)
        if st.session_state.anki_data:
            st.success("✅ Flashcards generated successfully!")
            
            # Download as CSV (Excel compatible)
            st.download_button(
                label="💾 Download for Excel / Anki (.csv)",
                data=st.session_state.anki_data.encode("utf-8"),
                file_name="DocuMind_Flashcards.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_btn_anki_csv"
            )

            # Download as TXT (Opens in Notepad on Windows)
            st.download_button(
                label="📄 Download for Notepad (.txt)",
                data=st.session_state.anki_data,
                file_name="DocuMind_Flashcards.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_btn_anki_txt"
            )

            with st.expander("👁️ Preview Flashcards on Screen", expanded=True):
                st.text_area("Flashcards Content", value=st.session_state.anki_data, height=180, disabled=True)

    # ----------------- SECTION 2: STUDY NOTES EXPORT -----------------
    with exp_col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-weight: 700; font-size: 1.1rem; color: #a5b4fc; margin-bottom: 2px;">
                📝 Export Complete Study Session
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 12px;">
                Save all your Q&A questions, answers, and page citations into clean, readable formats.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Build Notes Texts
        active_docs = list(set(st.session_state.indexed_docs))
        plain_notes = f"DOCUMIND AI — STUDY SESSION NOTES\nActive Documents: {', '.join(active_docs)}\n" + ("=" * 60) + "\n\n"
        
        html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DocuMind AI Study Notes</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; color: #24292f; }}
        h1 {{ border-bottom: 2px solid #0969da; padding-bottom: 8px; color: #0969da; }}
        .meta {{ color: #57609a; font-size: 0.9em; margin-bottom: 24px; }}
        .qa-card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }}
        .q-title {{ font-size: 1.1em; font-weight: bold; color: #1f2328; margin-bottom: 8px; }}
        .answer {{ color: #24292f; white-space: pre-wrap; }}
        .footer {{ text-align: center; font-size: 0.8em; color: #8c959f; margin-top: 40px; border-top: 1px solid #d0d7de; padding-top: 12px; }}
    </style>
</head>
<body>
    <h1>📚 DocuMind AI — Study Session Notes</h1>
    <div class="meta">Sources: {', '.join(active_docs)} | Date: Session Export</div>
"""

        if st.session_state.chat_history:
            for idx, t in enumerate(st.session_state.chat_history):
                plain_notes += f"Q{idx + 1}: {t['user']}\n\nANSWER:\n{t['assistant']}\n\n" + ("-" * 50) + "\n\n"
                clean_ans = t['assistant'].replace('<', '&lt;').replace('>', '&gt;')
                html_body += f"""
    <div class="qa-card">
        <div class="q-title">Q{idx + 1}: {t['user']}</div>
        <div class="answer">{clean_ans}</div>
    </div>
"""
        else:
            plain_notes += "No Q&A recorded yet in this session. Ask questions in the Q&A Reader first!\n"
            html_body += "<p><em>No questions recorded yet in this session.</em></p>"

        html_body += """
    <div class="footer">Generated by DocuMind AI // Neural Intelligence Engine</div>
</body>
</html>"""

        st.session_state.exported_notes_text = plain_notes
        st.session_state.exported_notes_html = html_body

        st.write("**Choose your download format:**")

        # Format 1: HTML Document (Opens in Chrome / Edge directly with double click!)
        st.download_button(
            label="🌐 Download Web Document (.html) — Opens in Browser",
            data=st.session_state.exported_notes_html.encode("utf-8"),
            file_name="DocuMind_Study_Notes.html",
            mime="text/html",
            use_container_width=True,
            key="dl_btn_notes_html"
        )

        # Format 2: TXT Document (Opens in Windows Notepad directly with double click!)
        st.download_button(
            label="📄 Download Text File (.txt) — Opens in Notepad",
            data=st.session_state.exported_notes_text.encode("utf-8"),
            file_name="DocuMind_Study_Notes.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_btn_notes_txt"
        )

        # Format 3: Markdown Document
        st.download_button(
            label="📝 Download Markdown (.md)",
            data=st.session_state.exported_notes_text.encode("utf-8"),
            file_name="DocuMind_Study_Notes.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_btn_notes_md"
        )

        with st.expander("👁️ Preview Notes on Screen", expanded=False):
            st.text_area("Notes Preview", value=st.session_state.exported_notes_text, height=200, disabled=True)
