import streamlit as st
from core.ui_shared import init_app, render_sidebar

st.set_page_config(
    page_title="Exam & Deep Research — DocuMind",
    page_icon="📝",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">🎓 Academic Assessment & Autonomous Research</div>
    <div class="brand-heading">Exam Paper & Deep Research</div>
    <div class="brand-tagline">
        Generate structured university-style question papers with marking rubrics or synthesize publication-grade multi-hop research reports.
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.vector_store.chunks:
    st.info("💡 **Getting Started:** Upload your book in the sidebar, or click **'📑 Load Sample'** to use this tool.")
else:
    exam_sub1, exam_sub2 = st.tabs(["🎓 Full Exam Paper & Answer Key", "🔬 Multi-Hop Research Dossier"])

    with exam_sub1:
        st.markdown("#### 🎓 University Question Paper & Confidential Rubric")
        e_col1, e_col2, e_col3 = st.columns([2, 1, 1])
        with e_col1:
            e_topic = st.text_input("Exam Syllabus / Topic", placeholder="e.g. Full Book, Chapter 4, Concurrency", key="page_e_topic")
        with e_col2:
            e_diff = st.selectbox("Exam Difficulty", options=["Standard Undergraduate", "Challenging / Competitive Exam", "Introductory"], key="page_e_diff")
        with e_col3:
            st.write("")
            st.write("")
            gen_exam_btn = st.button("Generate Paper", use_container_width=True, key="page_gen_exam")

        if gen_exam_btn:
            with st.spinner("Formulating university-style exam paper and rubric..."):
                res = pipeline.generate_exam_paper(topic=e_topic, difficulty=e_diff)
                st.session_state.active_exam_paper = res

        if st.session_state.get("active_exam_paper"):
            st.markdown("---")
            d_col_a, d_col_b = st.columns([1, 1])
            with d_col_a:
                st.download_button(
                    label="📥 Download Exam Paper (.md)",
                    data=st.session_state.active_exam_paper["content"].encode("utf-8"),
                    file_name="DocuMind_Exam_Paper.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_exam_md"
                )
            with d_col_b:
                st.download_button(
                    label="📄 Download Exam Paper (.txt)",
                    data=st.session_state.active_exam_paper["content"].encode("utf-8"),
                    file_name="DocuMind_Exam_Paper.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_exam_txt"
                )
            st.markdown(st.session_state.active_exam_paper["content"])

    with exam_sub2:
        st.markdown("#### 🔬 Autonomous Multi-Hop Research Dossier")
        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            d_topic = st.text_input("Deep Research Subject", placeholder="e.g. Critical Section Problem & Synchronization Primitives", key="page_d_topic")
        with d_col2:
            st.write("")
            st.write("")
            gen_dossier_btn = st.button("Generate Dossier", use_container_width=True, key="page_gen_dossier")

        if gen_dossier_btn:
            with st.spinner("Synthesizing multi-hop academic research report across chapters..."):
                res = pipeline.generate_research_dossier(topic=d_topic)
                st.session_state.active_dossier = res

        if st.session_state.get("active_dossier"):
            st.markdown("---")
            rd_col_a, rd_col_b = st.columns([1, 1])
            with rd_col_a:
                st.download_button(
                    label="📥 Download Research Dossier (.md)",
                    data=st.session_state.active_dossier["content"].encode("utf-8"),
                    file_name="DocuMind_Research_Dossier.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_dossier_md"
                )
            with rd_col_b:
                st.download_button(
                    label="📄 Download Research Dossier (.txt)",
                    data=st.session_state.active_dossier["content"].encode("utf-8"),
                    file_name="DocuMind_Research_Dossier.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_dossier_txt"
                )
            st.markdown(st.session_state.active_dossier["content"])
