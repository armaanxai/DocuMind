import streamlit as st
from core.ui_shared import init_app, render_sidebar

st.set_page_config(
    page_title="Mind Map & Summaries — DocuMind",
    page_icon="🧠",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">🧠 Structural Concept Analysis</div>
    <div class="brand-heading">Mind Maps & Synthesis</div>
    <div class="brand-tagline">
        Visualize knowledge relationships with interactive Mermaid diagrams, extract 1-click chapter cheat sheets, or cross-compare multiple documents.
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.vector_store.chunks:
    st.info("💡 **Getting Started:** Upload your book in the sidebar, or click **'📑 Load Sample'** to use this tool.")
else:
    sub1, sub2, sub3 = st.tabs([
        "🧠 Concept Mind Map",
        "📑 Chapter Cheat Sheet",
        "⚖️ Multi-Doc Comparison"
    ])

    with sub1:
        st.markdown("#### 🧠 Interactive Concept Mind Map (Mermaid Diagram)")
        st.caption("Visual graph showing how chapters, algorithms, and key definitions connect across the book.")
        
        mm_col1, mm_col2 = st.columns([3, 1])
        with mm_col1:
            mm_topic = st.text_input("Central Topic (Optional)", placeholder="e.g. Operating Systems Architecture, Attention, Concurrency", key="page_mm_topic")
        with mm_col2:
            st.write("")
            st.write("")
            gen_mm_btn = st.button("⚡ Generate Mind Map", use_container_width=True, key="page_gen_mm")

        if gen_mm_btn:
            with st.spinner("Extracting conceptual entities & generating graph hierarchy..."):
                res = pipeline.generate_mind_map(topic=mm_topic)
                st.session_state.active_mind_map = res

        if st.session_state.get("active_mind_map"):
            st.markdown("---")
            mm_d1, mm_d2 = st.columns([1, 1])
            with mm_d1:
                st.download_button(
                    label="📥 Download Mind Map (.mmd)",
                    data=st.session_state.active_mind_map['mermaid'].encode("utf-8"),
                    file_name="DocuMind_Mind_Map.mmd",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_mm_mmd"
                )
            with mm_d2:
                st.download_button(
                    label="📄 Download as Markdown (.md)",
                    data=f"# Mind Map\n\n```mermaid\n{st.session_state.active_mind_map['mermaid']}\n```".encode("utf-8"),
                    file_name="DocuMind_Mind_Map.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_mm_md"
                )
            st.markdown(f"```mermaid\n{st.session_state.active_mind_map['mermaid']}\n```")

    with sub2:
        st.markdown("#### 📑 1-Click Executive Summary & Exam Cheat Sheet")
        st.caption("Generate a high-yield summary, core principles, key definitions table, and essential takeaways from any chapter or topic.")

        sum_col1, sum_col2 = st.columns([3, 1])
        with sum_col1:
            sum_topic = st.text_input("Chapter or Topic to Summarize (Optional)", placeholder="e.g. Chapter 4, Threads, Virtual Memory, Attention Mechanism", key="page_sum_topic")
        with sum_col2:
            st.write("")
            st.write("")
            gen_sum_btn = st.button("⚡ Generate Cheat Sheet", use_container_width=True, key="page_gen_sum")

        if gen_sum_btn:
            with st.spinner("Synthesizing comprehensive summary and key takeaways..."):
                res = pipeline.generate_summary(topic=sum_topic)
                st.session_state.active_summary = res

        if st.session_state.get("active_summary"):
            st.markdown("---")
            s_d1, s_d2 = st.columns([1, 1])
            with s_d1:
                st.download_button(
                    label="📥 Download Cheat Sheet (.md)",
                    data=st.session_state.active_summary["content"].encode("utf-8"),
                    file_name="DocuMind_Cheat_Sheet.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_sum_md"
                )
            with s_d2:
                st.download_button(
                    label="📄 Download Cheat Sheet (.txt)",
                    data=st.session_state.active_summary["content"].encode("utf-8"),
                    file_name="DocuMind_Cheat_Sheet.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_sum_txt"
                )
            st.markdown(st.session_state.active_summary["content"])

    with sub3:
        st.markdown("#### ⚖️ Multi-Document Cross-Comparison Matrix")
        st.caption("Compare how multiple documents, chapters, or textbook editions explain the same core concept in a structured comparison matrix.")

        cmp_col1, cmp_col2 = st.columns([3, 1])
        with cmp_col1:
            cmp_topic = st.text_input("Comparison Topic", placeholder="e.g. Process vs Thread, Paging vs Segmentation, Self-Attention vs RNNs", key="page_cmp_topic")
        with cmp_col2:
            st.write("")
            st.write("")
            gen_cmp_btn = st.button("⚡ Compare Sections", use_container_width=True, key="page_gen_cmp")

        if gen_cmp_btn:
            with st.spinner("Analyzing contrasting viewpoints and synthesizing comparison matrix..."):
                res = pipeline.compare_documents(topic=cmp_topic)
                st.session_state.active_comparison = res

        if st.session_state.get("active_comparison"):
            st.markdown("---")
            c_d1, c_d2 = st.columns([1, 1])
            with c_d1:
                st.download_button(
                    label="📥 Download Comparison Matrix (.md)",
                    data=st.session_state.active_comparison["content"].encode("utf-8"),
                    file_name="DocuMind_Comparison.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_cmp_md"
                )
            with c_d2:
                st.download_button(
                    label="📄 Download Comparison Matrix (.txt)",
                    data=st.session_state.active_comparison["content"].encode("utf-8"),
                    file_name="DocuMind_Comparison.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_cmp_txt"
                )
            st.markdown(st.session_state.active_comparison["content"])
