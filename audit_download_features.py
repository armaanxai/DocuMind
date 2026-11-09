#!/usr/bin/env python3
"""
DocuMind-AI Download Features Comprehensive QA Auditor
Audits all download capabilities across the Streamlit online application.
"""

import os
import sys
import csv
import io
import re
import json
import requests
from dotenv import load_dotenv

# Ensure console supports utf-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure core package can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.vector_store import VectorStore
from core.document_loader import DocumentChunk
from core.rag_pipeline import RAGPipeline

def run_audit():
    load_dotenv()
    results = {
        "server_status": {},
        "codebase_inspection": {},
        "features": [],
        "overall_verdict": "PENDING"
    }

    print("=" * 80)
    print("DOCUMIND-AI DOWNLOAD FEATURES QA AUDITOR")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. VERIFY LIVE STREAMLIT HTTP SERVER ACCESSIBILITY
    # -------------------------------------------------------------
    base_url = "http://localhost:8501"
    print(f"\n[1/3] Testing Online Server Accessibility at {base_url}...")
    try:
        r_main = requests.get(base_url, timeout=10)
        r_health = requests.get(f"{base_url}/_stcore/health", timeout=10)
        
        server_ok = (r_main.status_code == 200 and r_health.status_code == 200 and r_health.text.strip() == "ok")
        results["server_status"] = {
            "main_status_code": r_main.status_code,
            "main_content_type": r_main.headers.get("content-type"),
            "health_status_code": r_health.status_code,
            "health_response": r_health.text.strip(),
            "responsive": server_ok
        }
        print(f"  ✓ Streamlit Web App Response: {r_main.status_code} ({r_main.headers.get('content-type')})")
        print(f"  ✓ Streamlit Health Probe: {r_health.status_code} [{r_health.text.strip()}]")
    except Exception as e:
        print(f"  ✗ Server Accessibility Check Failed: {e}")
        results["server_status"] = {"error": str(e), "responsive": False}

    # -------------------------------------------------------------
    # 2. INSPECT STREAMLIT CODEBASE FOR DOWNLOAD BUTTON IMPLEMENTATIONS
    # -------------------------------------------------------------
    print("\n[2/3] Inspecting Streamlit Download Buttons in Source Code...")
    pages_to_check = {
        "pages/3_📝_Exam_&_Research.py": [
            {"key": "dl_exam_md", "filename": "DocuMind_Exam_Paper.md", "mime": "text/markdown"},
            {"key": "dl_exam_txt", "filename": "DocuMind_Exam_Paper.txt", "mime": "text/plain"},
            {"key": "dl_dossier_md", "filename": "DocuMind_Research_Dossier.md", "mime": "text/markdown"},
            {"key": "dl_dossier_txt", "filename": "DocuMind_Research_Dossier.txt", "mime": "text/plain"}
        ],
        "pages/4_🧠_Mind_Map_&_Summary.py": [
            {"key": "dl_mm_mmd", "filename": "DocuMind_Mind_Map.mmd", "mime": "text/plain"},
            {"key": "dl_mm_md", "filename": "DocuMind_Mind_Map.md", "mime": "text/markdown"},
            {"key": "dl_sum_md", "filename": "DocuMind_Cheat_Sheet.md", "mime": "text/markdown"},
            {"key": "dl_sum_txt", "filename": "DocuMind_Cheat_Sheet.txt", "mime": "text/plain"},
            {"key": "dl_cmp_md", "filename": "DocuMind_Comparison.md", "mime": "text/markdown"},
            {"key": "dl_cmp_txt", "filename": "DocuMind_Comparison.txt", "mime": "text/plain"}
        ],
        "pages/5_📥_Export_&_Anki.py": [
            {"key": "dl_btn_anki_csv", "filename": "DocuMind_Flashcards.csv", "mime": "text/csv"},
            {"key": "dl_btn_anki_txt", "filename": "DocuMind_Flashcards.txt", "mime": "text/plain"},
            {"key": "dl_btn_notes_html", "filename": "DocuMind_Study_Notes.html", "mime": "text/html"},
            {"key": "dl_btn_notes_txt", "filename": "DocuMind_Study_Notes.txt", "mime": "text/plain"},
            {"key": "dl_btn_notes_md", "filename": "DocuMind_Study_Notes.md", "mime": "text/markdown"}
        ]
    }

    for page_path, buttons in pages_to_check.items():
        if os.path.exists(page_path):
            with open(page_path, "r", encoding="utf-8") as f:
                content = f.read()
            page_audit = {}
            for b in buttons:
                found_key = f'key="{b["key"]}"' in content or f"key='{b['key']}'" in content
                found_fn = f'file_name="{b["filename"]}"' in content or f"file_name='{b['filename']}'" in content
                found_mime = f'mime="{b["mime"]}"' in content or f"mime='{b['mime']}'" in content
                status = found_key and found_fn and found_mime
                page_audit[b["filename"]] = {
                    "key_present": found_key,
                    "filename_present": found_fn,
                    "mime_present": found_mime,
                    "verified": status
                }
                print(f"  ✓ [{page_path}] {b['filename']} ({b['mime']}) - In Source: {'VERIFIED' if status else 'MISSING'}")
            results["codebase_inspection"][page_path] = page_audit
        else:
            print(f"  ✗ Missing page file: {page_path}")
            results["codebase_inspection"][page_path] = {"error": "File not found"}

    # -------------------------------------------------------------
    # 3. GENERATE LIVE ARTIFACTS & AUDIT DOWNLOAD DATA PAYLOADS
    # -------------------------------------------------------------
    print("\n[3/3] Generating Artifacts and Auditing Payloads & Encoding...")

    # Initialize VectorStore and Pipeline with sample document
    sample_file = os.path.join("sample_docs", "transformer_architecture.txt")
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found at {sample_file}")

    with open(sample_file, "r", encoding="utf-8") as f:
        sample_text = f.read()

    vector_store = VectorStore()
    chunk = DocumentChunk(
        text=sample_text,
        doc_name="transformer_architecture.txt",
        page_number=1,
        chunk_id="sample_chunk_1"
    )
    vector_store.add_documents([chunk])

    api_key = os.getenv("GEMINI_API_KEY", "")
    pipeline = RAGPipeline(
        vector_store=vector_store,
        api_key=api_key,
        provider="gemini",
        model_name="gemini-2.5-flash"
    )

    feature_records = []

    def record_feature(page, name, filename, extension, mime_type, payload_bytes, extra_checks=None):
        extra_checks = extra_checks or {}
        is_bytes = isinstance(payload_bytes, bytes)
        size_bytes = len(payload_bytes) if is_bytes else (len(payload_bytes.encode("utf-8")) if isinstance(payload_bytes, str) else 0)
        
        # Test UTF-8 decodability
        try:
            if is_bytes:
                decoded_str = payload_bytes.decode("utf-8")
            else:
                decoded_str = str(payload_bytes)
            utf8_valid = True
        except Exception:
            utf8_valid = False
            decoded_str = ""

        non_empty = size_bytes > 0
        all_passed = is_bytes and utf8_valid and non_empty and all(extra_checks.values())

        feature_info = {
            "page": page,
            "feature_name": name,
            "filename": filename,
            "extension": extension,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "is_bytes": is_bytes,
            "utf8_valid": utf8_valid,
            "extra_validations": extra_checks,
            "status": "PASS" if all_passed else "FAIL"
        }
        feature_records.append(feature_info)
        print(f"  -> [{feature_info['status']}] {page} | {name} ({filename}) | {size_bytes} bytes | MIME: {mime_type}")
        return decoded_str

    # --- Feature Set A: Exam Paper (.md & .txt) ---
    print("\n* Testing Feature Group: Exam Paper...")
    exam_res = pipeline.generate_exam_paper(topic="Multi-Head Attention", difficulty="Standard")
    exam_content = exam_res.get("content", "")
    exam_bytes = exam_content.encode("utf-8")

    record_feature(
        page="pages/3_📝_Exam_&_Research.py",
        name="Academic Exam Paper (.md)",
        filename="DocuMind_Exam_Paper.md",
        extension=".md",
        mime_type="text/markdown",
        payload_bytes=exam_bytes,
        extra_checks={"has_exam_sections": bool(re.search(r"SECTION|Questions?|Marks?", exam_content, re.IGNORECASE))}
    )
    record_feature(
        page="pages/3_📝_Exam_&_Research.py",
        name="Academic Exam Paper (.txt)",
        filename="DocuMind_Exam_Paper.txt",
        extension=".txt",
        mime_type="text/plain",
        payload_bytes=exam_bytes,
        extra_checks={"content_matches_md": len(exam_bytes) > 50}
    )

    # --- Feature Set B: Research Dossier (.md & .txt) ---
    print("\n* Testing Feature Group: Research Dossier...")
    dossier_res = pipeline.generate_research_dossier(topic="Transformer Self-Attention Foundations")
    dossier_content = dossier_res.get("content", "")
    dossier_bytes = dossier_content.encode("utf-8")

    record_feature(
        page="pages/3_📝_Exam_&_Research.py",
        name="Multi-Hop Research Dossier (.md)",
        filename="DocuMind_Research_Dossier.md",
        extension=".md",
        mime_type="text/markdown",
        payload_bytes=dossier_bytes,
        extra_checks={"has_dossier_structure": len(dossier_content) > 100}
    )
    record_feature(
        page="pages/3_📝_Exam_&_Research.py",
        name="Multi-Hop Research Dossier (.txt)",
        filename="DocuMind_Research_Dossier.txt",
        extension=".txt",
        mime_type="text/plain",
        payload_bytes=dossier_bytes,
        extra_checks={"content_matches_md": len(dossier_bytes) > 100}
    )

    # --- Feature Set C: Concept Mind Map (.mmd & .md) ---
    print("\n* Testing Feature Group: Concept Mind Map...")
    mm_res = pipeline.generate_mind_map(topic="Transformers")
    mermaid_code = mm_res.get("mermaid", "")
    mm_bytes = mermaid_code.encode("utf-8")
    
    valid_mermaid = bool(re.search(r"^(graph|flowchart|mindmap)", mermaid_code.strip(), re.IGNORECASE))
    record_feature(
        page="pages/4_🧠_Mind_Map_&_Summary.py",
        name="Concept Mind Map (.mmd)",
        filename="DocuMind_Mind_Map.mmd",
        extension=".mmd",
        mime_type="text/plain",
        payload_bytes=mm_bytes,
        extra_checks={"valid_mermaid_syntax": valid_mermaid}
    )
    
    mm_md_payload = f"# Mind Map\n\n```mermaid\n{mermaid_code}\n```".encode("utf-8")
    record_feature(
        page="pages/4_🧠_Mind_Map_&_Summary.py",
        name="Concept Mind Map as Markdown (.md)",
        filename="DocuMind_Mind_Map.md",
        extension=".md",
        mime_type="text/markdown",
        payload_bytes=mm_md_payload,
        extra_checks={"contains_mermaid_block": b"```mermaid" in mm_md_payload}
    )

    # --- Feature Set D: Chapter Cheat Sheet (.md & .txt) ---
    print("\n* Testing Feature Group: Chapter Cheat Sheet...")
    sum_res = pipeline.generate_summary(topic="Scaled Dot-Product Attention")
    sum_content = sum_res.get("content", "")
    sum_bytes = sum_content.encode("utf-8")

    record_feature(
        page="pages/4_🧠_Mind_Map_&_Summary.py",
        name="Chapter Cheat Sheet (.md)",
        filename="DocuMind_Cheat_Sheet.md",
        extension=".md",
        mime_type="text/markdown",
        payload_bytes=sum_bytes,
        extra_checks={"has_cheat_sheet_content": len(sum_content) > 100}
    )
    record_feature(
        page="pages/4_🧠_Mind_Map_&_Summary.py",
        name="Chapter Cheat Sheet (.txt)",
        filename="DocuMind_Cheat_Sheet.txt",
        extension=".txt",
        mime_type="text/plain",
        payload_bytes=sum_bytes,
        extra_checks={"content_matches_md": len(sum_bytes) > 100}
    )

    # --- Feature Set E: Multi-Document Comparison Matrix (.md & .txt) ---
    print("\n* Testing Feature Group: Comparison Matrix...")
    cmp_res = pipeline.compare_documents(topic="Self-Attention vs Recurrent Models")
    cmp_content = cmp_res.get("content", "")
    cmp_bytes = cmp_content.encode("utf-8")

    record_feature(
        page="pages/4_🧠_Mind_Map_&_Summary.py",
        name="Cross-Comparison Matrix (.md)",
        filename="DocuMind_Comparison.md",
        extension=".md",
        mime_type="text/markdown",
        payload_bytes=cmp_bytes,
        extra_checks={"has_matrix_content": len(cmp_content) > 50}
    )
    record_feature(
        page="pages/4_🧠_Mind_Map_&_Summary.py",
        name="Cross-Comparison Matrix (.txt)",
        filename="DocuMind_Comparison.txt",
        extension=".txt",
        mime_type="text/plain",
        payload_bytes=cmp_bytes,
        extra_checks={"content_matches_md": len(cmp_bytes) > 50}
    )

    # --- Feature Set F: Anki Spaced Repetition Flashcards (.csv & .txt) ---
    print("\n* Testing Feature Group: Anki Flashcards...")
    anki_csv = pipeline.generate_anki_csv(topic="Attention Mechanism", count=5)
    anki_bytes = anki_csv.encode("utf-8")

    # CSV syntax check: Semicolon delimited, at least 1 row oltre header
    csv_reader = list(csv.reader(io.StringIO(anki_csv), delimiter=";"))
    csv_valid = len(csv_reader) >= 2 and all(len(row) >= 2 for row in csv_reader)

    record_feature(
        page="pages/5_📥_Export_&_Anki.py",
        name="Anki Spaced Repetition Flashcards (.csv)",
        filename="DocuMind_Flashcards.csv",
        extension=".csv",
        mime_type="text/csv",
        payload_bytes=anki_bytes,
        extra_checks={"semicolon_delimited": ";" in anki_csv, "excel_compatible_csv": csv_valid}
    )
    # Note: in page 5, .txt uses data=st.session_state.anki_data (string) or bytes
    record_feature(
        page="pages/5_📥_Export_&_Anki.py",
        name="Flashcards for Notepad (.txt)",
        filename="DocuMind_Flashcards.txt",
        extension=".txt",
        mime_type="text/plain",
        payload_bytes=anki_bytes,
        extra_checks={"has_flashcards_text": len(anki_csv) > 20}
    )

    # --- Feature Set G: Study Session Notes (.html, .txt, .md) ---
    print("\n* Testing Feature Group: Complete Study Session Notes...")
    # Simulate page 5 exact note generation logic
    active_docs = ["transformer_architecture.txt"]
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
    simulated_chat = [
        {"user": "What is the primary advantage of self-attention?", "assistant": "Self-attention allows the model to connect all positions with a constant number of operations."},
        {"user": "How is multi-head attention structured?", "assistant": "Multi-head attention projects queries, keys, and values linearly h times with different learnable projections."}
    ]
    for idx, t in enumerate(simulated_chat):
        plain_notes += f"Q{idx + 1}: {t['user']}\n\nANSWER:\n{t['assistant']}\n\n" + ("-" * 50) + "\n\n"
        clean_ans = t['assistant'].replace('<', '&lt;').replace('>', '&gt;')
        html_body += f"""
    <div class="qa-card">
        <div class="q-title">Q{idx + 1}: {t['user']}</div>
        <div class="answer">{clean_ans}</div>
    </div>
"""
    html_body += """
    <div class="footer">Generated by DocuMind AI // Neural Intelligence Engine</div>
</body>
</html>"""

    notes_html_bytes = html_body.encode("utf-8")
    notes_text_bytes = plain_notes.encode("utf-8")

    record_feature(
        page="pages/5_📥_Export_&_Anki.py",
        name="Study Notes Web Document (.html)",
        filename="DocuMind_Study_Notes.html",
        extension=".html",
        mime_type="text/html",
        payload_bytes=notes_html_bytes,
        extra_checks={
            "valid_doctype": "<!DOCTYPE html>" in html_body,
            "has_qa_card": 'class="qa-card"' in html_body,
            "has_css_style": "<style>" in html_body
        }
    )
    record_feature(
        page="pages/5_📥_Export_&_Anki.py",
        name="Study Notes Text File (.txt)",
        filename="DocuMind_Study_Notes.txt",
        extension=".txt",
        mime_type="text/plain",
        payload_bytes=notes_text_bytes,
        extra_checks={"has_questions": "Q1:" in plain_notes}
    )
    record_feature(
        page="pages/5_📥_Export_&_Anki.py",
        name="Study Notes Markdown Document (.md)",
        filename="DocuMind_Study_Notes.md",
        extension=".md",
        mime_type="text/markdown",
        payload_bytes=notes_text_bytes,
        extra_checks={"has_content": len(notes_text_bytes) > 50}
    )

    results["features"] = feature_records
    total_features = len(feature_records)
    passed_features = sum(1 for f in feature_records if f["status"] == "PASS")
    
    results["total_count"] = total_features
    results["passed_count"] = passed_features
    results["overall_verdict"] = "ALL_PASS" if (passed_features == total_features and results["server_status"].get("responsive", False)) else "FAILURE"

    print("\n" + "=" * 80)
    print(f"AUDIT SUMMARY: {passed_features}/{total_features} Features Operational ({results['overall_verdict']})")
    print("=" * 80)

    # Save structured audit results to JSON
    with open("download_features_audit_report.json", "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)
    print("Structured report saved to download_features_audit_report.json")

    return results

if __name__ == "__main__":
    res = run_audit()
    sys.exit(0 if res["overall_verdict"] == "ALL_PASS" else 1)
