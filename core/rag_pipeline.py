import time
import json
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from .vector_store import VectorStore
from .document_loader import DocumentChunk

# Universal LLM Client (OpenAI SDK for OpenAI, Groq, OpenRouter, DeepSeek, Local/Custom)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai
    from google.genai import types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

STRICT_BOOK_SYSTEM_PROMPT = """You are DocuMind, an ultra-strict Book Analysis and Verification Engine.
Your absolute duty is to extract and answer questions using ONLY the verbatim wording, direct sentences, and exact facts found in the provided Document/Book context.

MANDATORY RULES:
1. STRICT SOURCE GROUNDING: You must NEVER invent, assume, extrapolate, or use outside training knowledge. If it is not explicitly written in the provided context, state: "⚠️ यह जानकारी / उत्तर अपलोड की गई बुक या डॉक्यूमेंट में मौजूद नहीं है।"
2. EXACT BOOK WORDING (VERBATIM): For every point you make, you MUST quote the exact line or phrase from the source document inside quotation marks, followed by the source reference.
3. STRUCTURE YOUR ANSWER AS FOLLOWS:
   - 📖 **Exact Book Excerpt(s):** Quote the direct sentence(s) verbatim from the text that address the query.
   - 📝 **Direct Answer:** Synthesize the answer strictly adhering to the author's exact vocabulary and reasoning.
   - 📍 **Source Citation:** [Document Name, Page Number]
4. Maintain a formal, academic, and 100% faithful tone.
"""

class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str = None,
        provider: str = "auto",
        model_name: str = None,
        base_url: str = None
    ):
        self.vector_store = vector_store
        self.api_key = api_key
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.client = None
        self.openai_client = None

        if api_key:
            self.set_api_key(api_key, provider=provider, model_name=model_name, base_url=base_url)

    @staticmethod
    def detect_provider(api_key: str, user_provider: str = "auto") -> Dict[str, str]:
        """Auto-detects the provider, default model, and base URL from API key syntax."""
        key = (api_key or "").strip()
        p = (user_provider or "auto").lower()

        # Explicit user selection overrides
        if p and p != "auto":
            if "groq" in p:
                return {"provider": "groq", "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"}
            elif "openrouter" in p:
                return {"provider": "openrouter", "model": "openai/gpt-4o-mini", "base_url": "https://openrouter.ai/api/v1"}
            elif "openai" in p:
                return {"provider": "openai", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"}
            elif "deepseek" in p:
                return {"provider": "deepseek", "model": "deepseek-chat", "base_url": "https://api.deepseek.com"}
            elif "anthropic" in p:
                return {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "base_url": "https://api.anthropic.com"}
            elif "gemini" in p:
                return {"provider": "gemini", "model": "gemini-3.5-flash", "base_url": ""}

        # Intelligent key prefix detection
        if key.startswith("gsk_"):
            return {"provider": "groq", "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"}
        elif key.startswith("sk-or-"):
            return {"provider": "openrouter", "model": "openai/gpt-4o-mini", "base_url": "https://openrouter.ai/api/v1"}
        elif key.startswith("sk-ant-"):
            return {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "base_url": "https://api.anthropic.com"}
        elif key.startswith("AIzaSy") or key.startswith("AQ.") or key.startswith("ya29."):
            return {"provider": "gemini", "model": "gemini-3.5-flash", "base_url": ""}
        elif key.startswith("sk-") or key.startswith("sk-proj-"):
            return {"provider": "openai", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"}
        else:
            # Check if it has the format of a Google Gemini key
            if len(key) >= 30 and not key.startswith("sk-"):
                return {"provider": "gemini", "model": "gemini-3.5-flash", "base_url": ""}
            return {"provider": "openai", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"}

    def set_api_key(self, api_key: str, provider: str = "auto", model_name: str = None, base_url: str = None):
        """Universal initialization for any LLM API key."""
        if not api_key:
            return
        self.api_key = api_key.strip()
        info = self.detect_provider(self.api_key, user_provider=provider)
        self.provider = info["provider"]
        self.base_url = base_url or info["base_url"]
        self.model_name = model_name or info["model"]

        self.vector_store.set_api_key(self.api_key)

        # 1. OpenAI-compatible SDK initialization (OpenAI, Groq, OpenRouter, DeepSeek, Custom)
        if self.provider in ["groq", "openai", "openrouter", "deepseek", "custom"]:
            if HAS_OPENAI:
                try:
                    client_kwargs = {"api_key": self.api_key}
                    if self.base_url:
                        client_kwargs["base_url"] = self.base_url
                    if self.provider == "openrouter":
                        client_kwargs["default_headers"] = {
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "DocuMind"
                        }
                    self.openai_client = openai.OpenAI(**client_kwargs)
                except Exception:
                    self.openai_client = None
            else:
                self.openai_client = None

        # 2. Gemini fallback initialization
        elif self.provider == "gemini":
            if HAS_NEW_GENAI:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                except Exception:
                    self.client = None
            elif HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.api_key)
                except Exception:
                    pass

    def _call_gemini_internal(self, prompt: str, system_instruction: str = "", temperature: float = 0.2) -> str:
        """Fallback executor for Gemini."""
        if not self.api_key or not HAS_NEW_GENAI or not self.client:
            return "⚠️ Gemini API key is missing or SDK is unavailable."

        candidates = [self.model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
        seen = set()
        unique_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        last_error = None
        for model_id in unique_candidates:
            try:
                config_kwargs = {"temperature": temperature}
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction

                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                return response.text
            except Exception as e:
                last_error = e
                continue

        return f"Error executing generation: {str(last_error)}"

    def _call_llm(self, prompt: str, system_instruction: str = "", temperature: float = 0.2) -> str:
        """Universal LLM execution dispatcher supporting OpenAI, Groq, OpenRouter, DeepSeek, Anthropic & Gemini."""
        if not self.api_key:
            return "⚠️ API Key is missing. Please enter your API key to activate DocuMind."

        # 1. OpenAI-compatible / Groq / OpenRouter / DeepSeek / Custom
        if self.provider in ["groq", "openai", "openrouter", "deepseek", "custom"]:
            if self.openai_client:
                try:
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})

                    completion = self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature
                    )
                    return completion.choices[0].message.content or ""
                except Exception as e:
                    return f"Error executing generation with {self.provider.upper()}: {str(e)}"
            else:
                # Direct HTTP REST fallback via requests
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    if self.provider == "openrouter":
                        headers["HTTP-Referer"] = "http://localhost:8501"
                        headers["X-Title"] = "DocuMind"

                    url = f"{self.base_url.rstrip('/')}/chat/completions"
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})

                    payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": temperature
                    }
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        return f"API Error ({resp.status_code}): {resp.text}"
                except Exception as e:
                    return f"Connection error to {self.provider.upper()} API: {str(e)}"

        # 2. Anthropic Claude (HTTP REST)
        elif self.provider == "anthropic":
            try:
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": self.model_name or "claude-3-5-sonnet-20241022",
                    "max_tokens": 4096,
                    "system": system_instruction or STRICT_BOOK_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                resp = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["content"][0]["text"]
                else:
                    return f"Anthropic API Error ({resp.status_code}): {resp.text}"
            except Exception as e:
                return f"Anthropic connection error: {str(e)}"

        # 3. Gemini fallback
        elif self.provider == "gemini":
            return self._call_gemini_internal(prompt, system_instruction, temperature)

        return "⚠️ Unknown provider or unconfigured LLM engine."

    # Backward compatibility alias
    _call_gemini = _call_llm

    def _get_relevant_context(self, topic: str = "", top_k: int = 6) -> str:
        """Retrieves and formats representative book passages."""
        if not self.vector_store.chunks:
            return ""

        if topic.strip():
            retrieved = self.vector_store.similarity_search(topic.strip(), top_k=min(top_k, len(self.vector_store.chunks)))
            chunks_to_use = [chunk for chunk, _ in retrieved]
        else:
            total = len(self.vector_store.chunks)
            step = max(1, total // top_k)
            chunks_to_use = [self.vector_store.chunks[i] for i in range(0, total, step)][:top_k]

        return "\n\n".join([
            f"[Document: {c.doc_name} | Page {c.page_number}]:\n{c.text}"
            for c in chunks_to_use
        ])

    def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Strict verbatim grounded Q&A."""
        start_time = time.time()
        retrieved_results = self.vector_store.similarity_search(query, top_k=top_k)

        if not retrieved_results:
            return {
                "answer": "⚠️ No documents are currently loaded. Please upload a PDF book or click 'Load Sample'.",
                "sources": [],
                "latency_ms": 0,
                "model": self.model_name
            }

        context_blocks = []
        sources = []
        for chunk, score in retrieved_results:
            context_blocks.append(
                f"--- [Document: {chunk.doc_name} | Page: {chunk.page_number} | Score: {score:.2f}] ---\n{chunk.text}\n"
            )
            sources.append({
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "relevance_score": round(score, 3),
                "snippet": chunk.text[:220] + ("..." if len(chunk.text) > 220 else "")
            })

        user_content = f"""Book / Document Context:
{chr(10).join(context_blocks)}

User Query:
{query}

Formulate the answer strictly quoting the exact book wording above:"""

        answer_text = self._call_gemini(
            prompt=user_content,
            system_instruction=STRICT_BOOK_SYSTEM_PROMPT,
            temperature=0.1
        )
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "answer": answer_text,
            "sources": sources,
            "latency_ms": latency_ms,
            "model": self.model_name
        }

    def generate_summary(self, topic: str = "") -> Dict[str, Any]:
        """Feature 1: 1-Click Executive Summary & Cheat Sheet."""
        start_time = time.time()
        context = self._get_relevant_context(topic=topic, top_k=8)
        if not context:
            return {"content": "No document content available.", "latency_ms": 0}

        prompt = f"""Book Content Excerpts:
{context}

Topic / Chapter: {topic if topic else 'Overall Document'}

Task: Generate a high-yield Executive Summary & Revision Cheat Sheet:
1. 📌 **Executive Overview** (Brief 3-sentence summary of the chapter/topic)
2. 🔑 **Core Concepts & Principles** (Bullet points with verbatim quotes)
3. 📐 **Important Definitions & Key Terms** (Table of terms and their exact definitions from text)
4. ⚡ **Key Takeaways & Formulae** (Essential points to remember for exams)
Strictly ground everything in the text above. Cite [Page numbers]."""

        result = self._call_gemini(prompt, temperature=0.2)
        return {"content": result, "latency_ms": int((time.time() - start_time) * 1000)}

    def generate_interactive_quiz(self, topic: str = "", count: int = 4) -> List[Dict[str, Any]]:
        """Feature 2: Structured Interactive Quiz for live gamified testing."""
        context = self._get_relevant_context(topic=topic, top_k=6)
        if not context:
            return []

        prompt = f"""Book Content Excerpts:
{context}

Generate exactly {count} Multiple Choice Questions (MCQs) strictly derived from the book context.
Output ONLY a valid JSON list matching this structure:
[
  {{
    "question": "What is the primary characteristic of ...?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "According to page X: 'Quote from book'",
    "page": 1
  }}
]
Do not include any conversational preamble or text outside the JSON array."""

        raw = self._call_gemini(prompt, temperature=0.1)
        cleaned = re.sub(r'```json\s*', '', raw)
        cleaned = re.sub(r'```', '', cleaned).strip()

        try:
            quiz_data = json.loads(cleaned)
            if isinstance(quiz_data, list):
                return quiz_data
        except Exception:
            pass

        # Robust regex fallback to extract JSON array
        json_match = re.search(r'\[\s*\{.*\}\s*\]', raw, re.DOTALL)
        if json_match:
            try:
                quiz_data = json.loads(json_match.group(0))
                if isinstance(quiz_data, list) and len(quiz_data) > 0:
                    return quiz_data
            except Exception:
                pass

        # Fallback dummy question if JSON parsing fails
        return [{
            "question": f"Key concept regarding {topic if topic else 'the document'}:",
            "options": ["Stated directly in book", "External assumption", "Unrelated topic", "None of these"],
            "correct_index": 0,
            "explanation": "Grounded strictly in the uploaded text.",
            "page": 1
        }]

    def generate_mind_map(self, topic: str = "") -> Dict[str, Any]:
        """Feature 3: Interactive Mind Map & Concept Hierarchy in Mermaid.js."""
        start_time = time.time()
        context = self._get_relevant_context(topic=topic, top_k=7)
        if not context:
            return {"mermaid": "graph TD\n  Empty[No Documents Loaded]", "latency_ms": 0}

        prompt = f"""Book Content Excerpts:
{context}

Task: Create a visual concept Mind Map using Mermaid.js graph syntax (`graph TD`).
Rules:
- Identify the root subject and break it down into chapters/sub-concepts.
- Connect related ideas with labelled edges (e.g. A -->|implements| B).
- Keep node labels short (2-4 words) and wrap them in quotes, e.g. Node1["Multi-threading"].
- Output ONLY the mermaid diagram starting with `graph TD`. No extra markdown explanation."""

        raw = self._call_gemini(prompt, temperature=0.2)
        cleaned = re.sub(r'```mermaid\s*', '', raw)
        cleaned = re.sub(r'```', '', cleaned).strip()

        # If output has conversational preamble, extract starting from diagram keyword
        graph_match = re.search(r'(graph\s+(?:TD|TB|BT|RL|LR)|flowchart\s+(?:TD|TB|BT|RL|LR)|mindmap).*', cleaned, re.DOTALL | re.IGNORECASE)
        if graph_match:
            cleaned = graph_match.group(0).strip()
        elif not (cleaned.startswith("graph TD") or cleaned.startswith("graph LR") or cleaned.startswith("flowchart")):
            cleaned = "graph TD\n" + cleaned

        return {"mermaid": cleaned, "latency_ms": int((time.time() - start_time) * 1000)}

    def compare_documents(self, topic: str = "") -> Dict[str, Any]:
        """Feature 4: Multi-Document Cross-Comparison Matrix."""
        start_time = time.time()
        if not self.vector_store.chunks:
            return {"content": "No documents loaded for comparison.", "latency_ms": 0}

        # Check unique docs
        unique_docs = list(set(c.doc_name for c in self.vector_store.chunks))
        context = self._get_relevant_context(topic=topic, top_k=8)

        prompt = f"""Uploaded Documents in Knowledge Base: {', '.join(unique_docs)}
Context Excerpts:
{context}

Topic to Compare: {topic if topic else 'General Architecture & Methods'}

Task: Generate a thorough Multi-Document Cross-Comparison Analysis:
1. 📊 **Side-by-Side Comparison Matrix** (Markdown Table comparing Definition, Approach, Advantages, Limitations across the documents/sections).
2. 🔍 **Points of Agreement** (What both or all sections agree upon).
3. ⚡ **Contrasting Approaches & Nuances** (Differences in terminology or methodology).
4. 🏆 **Verdict & Recommendations** (When to apply which concept).
Strictly cite [Document Name, Page Number]."""

        result = self._call_gemini(prompt, temperature=0.2)
        return {"content": result, "latency_ms": int((time.time() - start_time) * 1000)}

    def generate_exam_paper(self, topic: str = "", difficulty: str = "Standard") -> Dict[str, Any]:
        """Feature 5: Full Academic Exam Paper with Answer Key & Rubric."""
        start_time = time.time()
        context = self._get_relevant_context(topic=topic, top_k=8)
        if not context:
            return {"content": "No documents loaded.", "latency_ms": 0}

        prompt = f"""Book Content Excerpts:
{context}

Topic: {topic if topic else 'Complete Syllabus'}
Difficulty: {difficulty}

Task: Create a professional Academic Question Paper:
- **SECTION A: Objective Assessment (4 MCQs / Short Questions - 1 Mark each)**
- **SECTION B: Conceptual Analytical Questions (3 Medium Questions - 3 Marks each)**
- **SECTION C: Comprehensive Applied / Case Study Question (1 Long Question - 5 Marks)**

---
### 🗝️ Confidential Answer Key & Grading Rubric:
Provide the exact verbatim book answer, marking breakdown, and page citations for every question."""

        result = self._call_gemini(prompt, temperature=0.2)
        return {"content": result, "latency_ms": int((time.time() - start_time) * 1000)}

    def generate_research_dossier(self, topic: str = "") -> Dict[str, Any]:
        """Feature 6: Multi-Hop Deep Research Report."""
        start_time = time.time()
        context = self._get_relevant_context(topic=topic, top_k=10)
        if not context:
            return {"content": "No documents loaded.", "latency_ms": 0}

        prompt = f"""Source Book Context:
{context}

Subject: {topic if topic else 'Core Systematic Analysis'}

Task: Formulate a comprehensive, publication-grade Academic Research Dossier:
1. **Abstract & Executive Summary**
2. **Background & Theoretical Foundations** (with verbatim quotes)
3. **Architectural / Systematic Breakdown**
4. **Critical Comparative Analysis**
5. **Practical Implications & Industry Applications**
6. **Academic References & Page Citations**
Ground strictly in the provided text without hallucination."""

        result = self._call_gemini(prompt, temperature=0.2)
        return {"content": result, "latency_ms": int((time.time() - start_time) * 1000)}

    def generate_anki_csv(self, topic: str = "", count: int = 8) -> str:
        """Feature 7: Export Anki Spaced Repetition Flashcards as CSV."""
        context = self._get_relevant_context(topic=topic, top_k=6)
        if not context:
            return "Front;Back;Source\nNo documents;No documents;N/A"

        prompt = f"""Book Context:
{context}

Generate {count} active recall flashcards strictly from the text.
Output format: Semicolon-separated CSV (`Front;Back;Source`).
- Front: A specific question or concept prompt.
- Back: The exact answer quoted or defined from the text.
- Source: Page number.
Output ONLY the CSV lines. No header, no markdown fences."""

        raw = self._call_gemini(prompt, temperature=0.1)
        cleaned = re.sub(r'```csv\s*', '', raw)
        cleaned = re.sub(r'```', '', cleaned).strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("front;back;source"):
            lines = lines[1:]
        return "Front;Back;Source\n" + "\n".join(lines)

    # Method Aliases for backward/external compatibility
    query = answer_query
    generate_cheat_sheet = generate_summary
    generate_deep_research = generate_research_dossier
    generate_mindmap = generate_mind_map
