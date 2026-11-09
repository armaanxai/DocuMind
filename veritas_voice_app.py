"""
VeritasVoice AI — Standalone Autonomous Multi-Agent Research Swarm & Voice Copilot
Flagship Application combining:
  1. Multi-Agent Autonomous Research & Fact-Verification Engine
  2. Privacy-First Local Voice AI Copilot & Speech Synthesis
"""

import os
import io
import csv
import streamlit as st

from core.ui_shared import init_app, render_sidebar, render_audio_player
from core.agent_copilot import AutonomousAgentCopilot, AutonomousMissionResult

st.set_page_config(
    page_title="VeritasVoice AI — Multi-Agent Research & Voice Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

pipeline = init_app()
render_sidebar()

# Auto-preload sample document if vector store is empty for instant zero-click out-of-the-box readiness
if not st.session_state.vector_store.chunks:
    sample_path = os.path.join("sample_docs", "transformer_architecture.txt")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_content = f.read()
        sample_chunks = st.session_state.loader.load_text(sample_content, doc_name="Transformer_Architecture_Paper.txt")
        st.session_state.vector_store.set_api_key(st.session_state.api_key)
        st.session_state.vector_store.add_documents(sample_chunks)
        if "Transformer_Architecture_Paper.txt" not in st.session_state.indexed_docs:
            st.session_state.indexed_docs.append("Transformer_Architecture_Paper.txt")

# Custom Glassmorphic Styling for VeritasVoice AI
st.markdown("""
<style>
    .veritas-hero {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.16) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(16, 185, 129, 0.10) 100%);
        backdrop-filter: blur(24px) saturate(190%);
        -webkit-backdrop-filter: blur(24px) saturate(190%);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.2);
    }
    .swarm-pipeline {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .swarm-node {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 16px 18px;
        text-align: center;
        backdrop-filter: blur(18px);
        transition: all 0.25s ease;
    }
    .swarm-node:hover {
        border-color: rgba(165, 180, 252, 0.45);
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
    }
    .swarm-node-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: #ffffff;
        margin-top: 8px;
    }
    .swarm-node-desc {
        font-size: 0.76rem;
        color: #94a3b8;
        margin-top: 3px;
    }
    .truth-gauge-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.14) 0%, rgba(16, 185, 129, 0.14) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 22px 28px;
        margin-bottom: 22px;
        backdrop-filter: blur(20px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
    }
    .truth-score-val {
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #34d399;
        line-height: 1;
    }
    .claim-card {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .claim-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(165, 180, 252, 0.3);
    }
    .veracity-badge {
        font-size: 0.74rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-verified {
        background: rgba(16, 185, 129, 0.22);
        border: 1px solid rgba(52, 211, 153, 0.55);
        color: #34d399;
    }
    .badge-partially {
        background: rgba(245, 158, 11, 0.22);
        border: 1px solid rgba(251, 191, 36, 0.55);
        color: #fbbf24;
    }
    .badge-contested {
        background: rgba(239, 68, 68, 0.22);
        border: 1px solid rgba(248, 113, 113, 0.55);
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="veritas-hero">
    <div class="pill-tag">🤖 Project 1 + 2 Flagship • Live on Localhost</div>
    <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-top: 8px; letter-spacing: -0.02em;">
        VeritasVoice AI
    </div>
    <div style="font-size: 1.05rem; font-weight: 600; color: #a5b4fc; margin-top: 4px;">
        Autonomous Multi-Agent Research Swarm & Conversational Voice Copilot
    </div>
    <div style="color: #cbd5e1; font-size: 0.90rem; margin-top: 10px; line-height: 1.6; max-width: 900px;">
        Coordinates an ensemble of 4 specialized AI agents (Planner, Researcher, Adversarial Critic, and Voice Narrator) to decompose complex topics, cross-examine evidence, compute an empirical <b>Truth Index (0–100%)</b>, and generate spoken voice briefings with verbatim source citations.
    </div>
</div>
""", unsafe_allow_html=True)

# 4-Agent Pipeline Overview
st.markdown("""
<div class="swarm-pipeline">
    <div class="swarm-node">
        <div style="font-size: 1.6rem;">🎯</div>
        <div class="swarm-node-title">1. Planner Agent</div>
        <div class="swarm-node-desc">Hypothesis Decomposition</div>
    </div>
    <div class="swarm-node">
        <div style="font-size: 1.6rem;">🔍</div>
        <div class="swarm-node-title">2. Researcher Agent</div>
        <div class="swarm-node-desc">Multi-Hop Vector Retrieval</div>
    </div>
    <div class="swarm-node">
        <div style="font-size: 1.6rem;">⚖️</div>
        <div class="swarm-node-title">3. Critic Agent</div>
        <div class="swarm-node-desc">Adversarial Fact-Check & Score</div>
    </div>
    <div class="swarm-node">
        <div style="font-size: 1.6rem;">🎙️</div>
        <div class="swarm-node-title">4. Voice Narrator</div>
        <div class="swarm-node-desc">Dossier & Speech Synthesis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "agent_copilot" not in st.session_state:
    st.session_state.agent_copilot = AutonomousAgentCopilot(
        pipeline=pipeline,
        vector_store=st.session_state.vector_store
    )

if "active_mission" not in st.session_state:
    st.session_state.active_mission = None

# Input Section with Quick-Start Suggestion Chips
st.markdown("#### 🚀 Launch an Autonomous Mission")
col_chips = st.columns(4)
with col_chips[0]:
    if st.button("⚡ Attention & BLEU Score", use_container_width=True, key="chip_attn"):
        st.session_state.target_mission_input = "Transformer Architecture and Scaled Dot-Product Attention BLEU score"
with col_chips[1]:
    if st.button("⚡ Self-Attention vs RNNs", use_container_width=True, key="chip_rnn"):
        st.session_state.target_mission_input = "Comparison of Multi-Head Self-Attention versus Recurrent Neural Networks"
with col_chips[2]:
    if st.button("⚡ Positional Encodings", use_container_width=True, key="chip_pos"):
        st.session_state.target_mission_input = "Sinusoidal Positional Encoding Mechanics and Sequence Order"
with col_chips[3]:
    if st.button("⚡ Encoder-Decoder Layers", use_container_width=True, key="chip_enc"):
        st.session_state.target_mission_input = "Encoder and Decoder Stacks N=6 layers and residual connections"

input_val = st.session_state.get("target_mission_input", "Transformer Architecture and Self-Attention Mechanism")

col_inp, col_btn = st.columns([4, 1])
with col_inp:
    mission_topic = st.text_input(
        "Enter any research subject or query:",
        value=input_val,
        placeholder="Type research topic...",
        key="main_mission_topic_input"
    )
with col_btn:
    st.write("")
    st.write("")
    launch_btn = st.button("⚡ Run Swarm", use_container_width=True, key="btn_run_swarm")

progress_bar = st.empty()
status_msg = st.empty()

if launch_btn and mission_topic.strip():
    def update_progress(msg: str, pct: int):
        progress_bar.progress(pct)
        status_msg.caption(f"⚡ {msg}")

    with st.spinner("Swarm agents collaborating in real-time..."):
        result: AutonomousMissionResult = st.session_state.agent_copilot.execute_mission(
            topic=mission_topic.strip(),
            progress_callback=update_progress
        )
        st.session_state.active_mission = result
    progress_bar.empty()
    status_msg.empty()

mission = st.session_state.get("active_mission")

if mission:
    st.markdown("---")
    badge_color = "#34d399" if mission.overall_truth_score >= 80 else ("#fbbf24" if mission.overall_truth_score >= 60 else "#f87171")
    
    # 1. Truth Index Gauge Card
    st.markdown(f"""
    <div class="truth-gauge-card">
        <div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em;">
                🛡️ Peer Review & Factuality Verification
            </div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin-top: 4px;">
                Grounding & Truth Index: <span style="color: {badge_color};">{mission.overall_truth_score}%</span>
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 3px;">
                Audited <b>{len(mission.verified_claims)}</b> discrete claims across <b>{len(mission.plan)}</b> research tracks in <b>{mission.latency_ms} ms</b>.
            </div>
        </div>
        <div class="truth-score-val" style="color: {badge_color};">
            {mission.overall_truth_score}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Spoken Voice Debrief & Audio Player
    st.markdown("### 🎙️ Spoken Voice Debrief (Hands-Free Listening)")
    st.caption("Agent 4 (Voice Narrator) synthesized this natural voice audio briefing from the verified research report.")

    render_audio_player(mission.voice_script, f"vv_voice_{mission.mission_id}")

    if mission.audio_wav_bytes:
        with st.expander("📻 Download & Play Dedicated .WAV Voice File", expanded=False):
            st.audio(mission.audio_wav_bytes, format="audio/wav")
            st.download_button(
                label="💾 Download Voice Track (.wav)",
                data=mission.audio_wav_bytes,
                file_name=f"VeritasVoice_Brief_{mission.mission_id}.wav",
                mime="audio/wav",
                use_container_width=True,
                key=f"dl_wav_{mission.mission_id}"
            )

    # 3. Tabs: Claims vs Full Report vs Swarm Execution Trace
    tab_claims, tab_report, tab_trace = st.tabs([
        "⚖️ Discrete Claim Verification Matrix",
        "📄 Full Verified Research Report",
        "⚡ Autonomous Swarm Trace"
    ])

    with tab_claims:
        st.markdown("#### 🔬 Peer-Reviewed Claims Audited by Adversarial Critic")
        for c in mission.verified_claims:
            b_class = "badge-verified" if c.veracity == "VERIFIED" else ("badge-partially" if c.veracity == "PARTIALLY_SUPPORTED" else "badge-contested")
            st.markdown(f"""
            <div class="claim-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="veracity-badge {b_class}">{c.veracity} ({c.confidence_score}%)</span>
                    <span style="font-size: 0.80rem; color: #94a3b8;">📄 {c.doc_name} • Page {c.source_page}</span>
                </div>
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.96rem; margin-bottom: 8px;">
                    #{c.claim_id}: {c.claim_text}
                </div>
                <div style="font-size: 0.84rem; color: #cbd5e1; font-style: italic; background: rgba(0,0,0,0.3); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #818cf8; margin-bottom: 8px;">
                    "{c.source_evidence}"
                </div>
                <div style="font-size: 0.80rem; color: #a5b4fc;">
                    <strong>Critic Audit Note:</strong> {c.critic_notes}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Export Truth Matrix as CSV
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Claim ID", "Claim Text", "Veracity", "Confidence", "Doc Name", "Page", "Evidence", "Critic Notes"])
        for c in mission.verified_claims:
            writer.writerow([c.claim_id, c.claim_text, c.veracity, f"{c.confidence_score}%", c.doc_name, c.source_page, c.source_evidence, c.critic_notes])

        st.download_button(
            label="📊 Download Truth Matrix (.csv)",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name=f"VeritasVoice_Truth_Matrix_{mission.mission_id}.csv",
            mime="text/csv",
            key="dl_truth_matrix_csv_main"
        )

    with tab_report:
        st.markdown("#### 📑 Publication-Grade Research Report")
        
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=mission.report_markdown.encode("utf-8"),
                file_name=f"VeritasVoice_Report_{mission.mission_id}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_report_md_main"
            )
        with rep_col2:
            st.download_button(
                label="📄 Download Research Report (.txt)",
                data=mission.report_markdown.encode("utf-8"),
                file_name=f"VeritasVoice_Report_{mission.mission_id}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_report_txt_main"
            )

        st.markdown("---")
        st.markdown(mission.report_markdown)

    with tab_trace:
        st.markdown("#### ⚡ Autonomous Agent Execution Trace")
        for t in mission.execution_trace:
            st.markdown(f"""
            <div class="inventory-item">
                <div class="inventory-title">
                    🤖 <b>{t['agent']}</b>
                </div>
                <div class="inventory-meta">
                    <span>⏱️ <b>{t['duration_ms']} ms</b></span>
                    <span>{t['output']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Helpful Onboarding state
    st.info("💡 **Ready to Research:** Select one of the quick-start suggestion chips above or type your own subject, then click **'⚡ Run Swarm'**.")
