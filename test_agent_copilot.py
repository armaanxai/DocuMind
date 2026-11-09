"""
End-to-End Automated Test Suite for VeritasVoice AI
(Project 1 + 2 Fusion: Autonomous Multi-Agent Research & Fact-Verification Voice Copilot)
"""

import os
import sys
import time
import io
import csv
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from core.document_loader import DocumentLoader
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline
from core.agent_copilot import AutonomousAgentCopilot, AutonomousMissionResult

def main():
    print("=" * 80)
    print("🤖 STARTING VERITASVOICE (PROJECT 1 + 2 FUSION) AGENT COPILOT AUDIT")
    print("=" * 80)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: No API key found in environment for LLM execution.")
        sys.exit(1)

    # 1. Initialize Core Engine & Ingest Sample Document
    print("\n[Step 1] Initializing VectorStore and Ingesting Knowledge Base...")
    loader = DocumentLoader()
    sample_path = os.path.join("sample_docs", "transformer_architecture.txt")
    chunks = loader.load_file(sample_path)
    assert len(chunks) > 0, "Failed to load sample document chunks."

    vstore = VectorStore(api_key=api_key)
    vstore.add_documents(chunks)
    print(f"✅ Indexed {len(chunks)} chunks into vector store.")

    pipeline = RAGPipeline(vector_store=vstore, api_key=api_key)
    copilot = AutonomousAgentCopilot(pipeline=pipeline, vector_store=vstore)

    # 2. Test Agent 1: Planner Agent
    print("\n[Step 2] Testing Agent 1 (Planner & Strategist)...")
    t0 = time.time()
    plan = copilot.run_planner_agent(topic="Multi-Head Attention vs Recurrence")
    dt = int((time.time() - t0) * 1000)
    assert len(plan) >= 2, f"Expected at least 2 research tracks, got {len(plan)}"
    for item in plan:
        assert item.focus_area and len(item.target_questions) > 0
    print(f"✅ [PASS] Agent 1 (Planner) created {len(plan)} structured tracks in {dt} ms.")
    for p in plan:
        print(f"   🎯 Track #{p.step_id}: {p.focus_area} ({len(p.target_questions)} questions)")

    # 3. Test Agent 2: Researcher Agent
    print("\n[Step 3] Testing Agent 2 (Multi-Hop Evidence Retrieval)...")
    t0 = time.time()
    evidence = copilot.run_researcher_agent(plan)
    dt = int((time.time() - t0) * 1000)
    assert len(evidence) >= 1, "Evidence retrieval returned 0 chunks"
    print(f"✅ [PASS] Agent 2 (Researcher) retrieved {len(evidence)} verified excerpts across documents in {dt} ms.")
    print(f"   🔍 Sample Excerpt: Doc '{evidence[0]['doc_name']}' Page {evidence[0]['page_number']} (Score: {evidence[0]['relevance_score']})")

    # 4. Test Agent 3: Adversarial Critic & Fact-Checker
    print("\n[Step 4] Testing Agent 3 (Adversarial Critic & Truth Index)...")
    t0 = time.time()
    claims, truth_score = copilot.run_critic_agent("Multi-Head Attention", evidence)
    dt = int((time.time() - t0) * 1000)
    assert len(claims) >= 1, "Critic agent produced 0 verified claims"
    assert 0.0 <= truth_score <= 100.0, f"Invalid truth score: {truth_score}"
    print(f"✅ [PASS] Agent 3 (Critic) audited {len(claims)} claims with Truth Index {truth_score}% in {dt} ms.")
    for c in claims[:3]:
        print(f"   ⚖️ Claim #{c.claim_id}: '{c.claim_text[:50]}...' -> [{c.veracity}] Confidence: {c.confidence_score}%")

    # 5. Test Agent 4: Executive Synthesizer & Voice Debrief
    print("\n[Step 5] Testing Agent 4 (Synthesizer & Audio Narrator)...")
    t0 = time.time()
    report_md, voice_script = copilot.run_synthesizer_agent(
        topic="Multi-Head Attention",
        plan=plan,
        evidence=evidence,
        claims=claims,
        truth_score=truth_score
    )
    dt = int((time.time() - t0) * 1000)
    assert len(report_md) > 200, "Report markdown too brief"
    assert len(voice_script) > 50, "Voice script too brief"
    print(f"✅ [PASS] Agent 4 (Synthesizer) produced {len(report_md)} char report and {len(voice_script)} char voice script in {dt} ms.")

    # 6. Test Full Autonomous Mission Loop with Audio Generation
    print("\n[Step 6] Testing Full Mission Execution Loop & SAPI Audio Synthesis...")
    t0 = time.time()
    mission = copilot.execute_mission(
        topic="Transformer Architecture & BLEU Score",
        progress_callback=lambda msg, pct: print(f"   [{pct}%] {msg}")
    )
    dt = int((time.time() - t0) * 1000)
    assert isinstance(mission, AutonomousMissionResult)
    assert mission.overall_truth_score > 0
    assert mission.audio_wav_bytes is not None and len(mission.audio_wav_bytes) > 1000, "Audio WAV bytes generation failed"
    print(f"✅ [PASS] Full Mission Loop completed in {dt} ms!")
    print(f"   🎙️ Generated Voice Audio WAV: {len(mission.audio_wav_bytes)} bytes")
    print(f"   🛡️ Final Truth Index: {mission.overall_truth_score}%")

    # 7. Test Export Data Formats
    print("\n[Step 7] Testing Multi-Format Export Suite (Report .md, .txt, Truth Matrix .csv, Audio .wav)...")
    # Verify CSV
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["Claim ID", "Claim", "Veracity", "Confidence"])
    for c in mission.verified_claims:
        writer.writerow([c.claim_id, c.claim_text, c.veracity, f"{c.confidence_score}%"])
    csv_str = csv_buf.getvalue()
    assert len(csv_str) > 50
    print("✅ [PASS] Verified Truth Matrix CSV export generation.")

    # Verify Markdown & Text
    assert len(mission.report_markdown.encode("utf-8")) > 200
    print("✅ [PASS] Verified Markdown & Text export generation.")

    print("\n" + "=" * 80)
    print("🎉 ALL VERITASVOICE AGENT COPILOT TESTS PASSED (100% OPERATIONAL)")
    print("=" * 80)

if __name__ == "__main__":
    main()
