import io
import csv
import streamlit as st
from core.ui_shared import init_app, render_sidebar, render_audio_player
from core.agent_copilot import AutonomousAgentCopilot, AutonomousMissionResult

st.set_page_config(
    page_title="Agent Voice Copilot — DocuMind",
    page_icon="🤖",
    layout="wide"
)

pipeline = init_app()
render_sidebar()

# Page CSS styling
st.markdown("""
<style>
    .swarm-pipeline {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    .swarm-node {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
        backdrop-filter: blur(16px);
        transition: all 0.25s ease;
    }
    .swarm-node:hover {
        border-color: rgba(165, 180, 252, 0.4);
        transform: translateY(-2px);
    }
    .swarm-node-title {
        font-weight: 700;
        font-size: 0.90rem;
        color: #ffffff;
        margin-top: 6px;
    }
    .swarm-node-desc {
        font-size: 0.74rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .truth-gauge-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(16, 185, 129, 0.12) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(20px);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .truth-score-val {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #34d399;
        line-height: 1;
    }
    .claim-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .claim-card:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    .veracity-badge {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-verified {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(52, 211, 153, 0.5);
        color: #34d399;
    }
    .badge-partially {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(251, 191, 36, 0.5);
        color: #fbbf24;
    }
    .badge-contested {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(248, 113, 113, 0.5);
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div class="pill-tag">🤖 Project 1 + 2 Fusion: VeritasVoice AI</div>
    <div class="brand-heading">Autonomous Agent Research & Voice Copilot</div>
    <div class="brand-tagline">
        An autonomous ensemble of Planner, Researcher, Adversarial Critic, and Voice Narrator agents collaborating to formulate verifiable, hallucination-audited research dossiers with live conversational audio debriefs.
    </div>
</div>
""", unsafe_allow_html=True)

# Visual Swarm Pipeline Architecture
st.markdown("""
<div class="swarm-pipeline">
    <div class="swarm-node">
        <div style="font-size: 1.4rem;">🎯</div>
        <div class="swarm-node-title">1. Planner Agent</div>
        <div class="swarm-node-desc">Hypothesis Decomposition</div>
    </div>
    <div class="swarm-node">
        <div style="font-size: 1.4rem;">🔍</div>
        <div class="swarm-node-title">2. Researcher Agent</div>
        <div class="swarm-node-desc">Multi-Hop Vector Retrieval</div>
    </div>
    <div class="swarm-node">
        <div style="font-size: 1.4rem;">⚖️</div>
        <div class="swarm-node-title">3. Critic Agent</div>
        <div class="swarm-node-desc">Adversarial Fact-Check & Score</div>
    </div>
    <div class="swarm-node">
        <div style="font-size: 1.4rem;">🎙️</div>
        <div class="swarm-node-title">4. Voice Narrator</div>
        <div class="swarm-node-desc">Dossier & Speech Synthesis</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.vector_store.chunks:
    st.info("💡 **Getting Started:** Upload textbooks/PDFs in the sidebar, or click **'📑 Load Sample'** to initialize the autonomous agent swarm.")
else:
    # Initialize Copilot session state
    if "agent_copilot" not in st.session_state:
        st.session_state.agent_copilot = AutonomousAgentCopilot(
            pipeline=pipeline,
            vector_store=st.session_state.vector_store
        )
    if "active_mission" not in st.session_state:
        st.session_state.active_mission = None

    c_input, c_btn = st.columns([4, 1])
    with c_input:
        default_topic = "Transformer Architecture and Self-Attention Mechanism"
        mission_topic = st.text_input(
            "Research Topic / Subject for Autonomous Swarm",
            value=default_topic,
            placeholder="e.g. Concurrency & Deadlocks, Transformer Attention, Raft Consensus, Quantum Computing",
            key="input_mission_topic"
        )
    with c_btn:
        st.write("")
        st.write("")
        launch_btn = st.button("🚀 Launch Mission", use_container_width=True, key="btn_launch_mission")

    progress_bar = st.empty()
    status_msg = st.empty()

    if launch_btn and mission_topic.strip():
        def update_progress(msg: str, pct: int):
            progress_bar.progress(pct)
            status_msg.caption(f"⚡ {msg}")

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
        
        # 1. Truth Index Gauge Card
        badge_color = "#34d399" if mission.overall_truth_score >= 80 else ("#fbbf24" if mission.overall_truth_score >= 60 else "#f87171")
        st.markdown(f"""
        <div class="truth-gauge-card">
            <div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em;">
                    🛡️ Peer Review & Factuality Verification
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 4px;">
                    Grounding & Truth Index: <span style="color: {badge_color};">{mission.overall_truth_score}%</span>
                </div>
                <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 2px;">
                    Audited {len(mission.verified_claims)} discrete claims across {len(mission.plan)} research tracks in {mission.latency_ms} ms.
                </div>
            </div>
            <div class="truth-score-val" style="color: {badge_color};">
                {mission.overall_truth_score}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Interactive Spoken Voice Debrief
        st.markdown("### 🎙️ Executive Voice Debrief & Audio Player")
        st.caption("Spoken audio summary synthesized by Agent 4 for instant hands-free listening.")
        
        # In-Browser Web Speech + HD Voice Player
        render_audio_player(mission.voice_script, f"mission_voice_{mission.mission_id}")
        
        if mission.audio_wav_bytes:
            with st.expander("📻 Download & Play Dedicated .WAV Voice Track", expanded=False):
                st.audio(mission.audio_wav_bytes, format="audio/wav")
                st.download_button(
                    label="💾 Download Voice Track (.wav)",
                    data=mission.audio_wav_bytes,
                    file_name=f"DocuMind_Voice_Brief_{mission.mission_id}.wav",
                    mime="audio/wav",
                    use_container_width=True,
                    key=f"dl_wav_{mission.mission_id}"
                )

        # 3. Tabs: Verified Claims vs Full Report vs Swarm Execution Trace
        tab_claims, tab_report, tab_trace = st.tabs([
            "⚖️ Verified Claim Matrix",
            "📄 Full Research Dossier",
            "⚡ Swarm Execution Trace"
        ])

        with tab_claims:
            st.markdown("#### 🔬 Discrete Empirical Claims Audited by Adversarial Critic")
            for c in mission.verified_claims:
                b_class = "badge-verified" if c.veracity == "VERIFIED" else ("badge-partially" if c.veracity == "PARTIALLY_SUPPORTED" else "badge-contested")
                st.markdown(f"""
                <div class="claim-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="veracity-badge {b_class}">{c.veracity} ({c.confidence_score}%)</span>
                        <span style="font-size: 0.78rem; color: #94a3b8;">📄 {c.doc_name} • Page {c.source_page}</span>
                    </div>
                    <div style="font-weight: 600; color: #f1f5f9; font-size: 0.94rem; margin-bottom: 6px;">
                        #{c.claim_id}: {c.claim_text}
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; font-style: italic; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 8px; border-left: 3px solid #818cf8; margin-bottom: 6px;">
                        "{c.source_evidence}"
                    </div>
                    <div style="font-size: 0.78rem; color: #a5b4fc;">
                        <strong>Critic Note:</strong> {c.critic_notes}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # CSV Export for Claims
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Claim ID", "Claim Text", "Veracity", "Confidence", "Doc Name", "Page", "Evidence", "Critic Notes"])
            for c in mission.verified_claims:
                writer.writerow([c.claim_id, c.claim_text, c.veracity, f"{c.confidence_score}%", c.doc_name, c.source_page, c.source_evidence, c.critic_notes])

            st.download_button(
                label="📊 Export Truth Matrix (.csv)",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name=f"DocuMind_Truth_Matrix_{mission.mission_id}.csv",
                mime="text/csv",
                key="dl_truth_matrix_csv"
            )

        with tab_report:
            st.markdown("#### 📑 Publication-Grade Research Report")
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="📥 Download Research Report (.md)",
                    data=mission.report_markdown.encode("utf-8"),
                    file_name=f"DocuMind_Research_Report_{mission.mission_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_report_md"
                )
            with d_col2:
                st.download_button(
                    label="📄 Download Research Report (.txt)",
                    data=mission.report_markdown.encode("utf-8"),
                    file_name=f"DocuMind_Research_Report_{mission.mission_id}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_report_txt"
                )

            st.markdown("---")
            st.markdown(mission.report_markdown)

        with tab_trace:
            st.markdown("#### ⚡ Autonomous Agent Execution Trace")
            for t in mission.execution_trace:
                st.markdown(f"""
                <div class="inventory-item">
                    <div class="inventory-title">
                        🤖 {t['agent']}
                    </div>
                    <div class="inventory-meta">
                        <span>⏱️ {t['duration_ms']} ms</span>
                        <span>{t['output']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
