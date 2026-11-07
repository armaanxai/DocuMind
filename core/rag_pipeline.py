import time
from typing import List, Dict, Any
from .vector_store import VectorStore
from .document_loader import DocumentChunk

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

RAG_SYSTEM_PROMPT = """You are DocuMind AI, an elite Research Assistant and Document Analyst.
Your mission is to provide accurate, grounded answers strictly based on the retrieved context below.

CRITICAL INSTRUCTIONS:
1. ONLY answer using facts directly supported by the provided Context Chunks.
2. If the context does not contain enough information to answer truthfully, state clearly: "The provided document does not contain sufficient details to answer this query."
3. CITATIONS: Always attribute your claims by citing the document name and page number, e.g., [Doc: Attention.pdf, Page 3].
4. Maintain a clear, concise, and professional tone with bullet points and structured markdown.
"""

class RAGPipeline:
    def __init__(self, vector_store: VectorStore, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.vector_store = vector_store
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        if api_key:
            self.set_api_key(api_key)

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.vector_store.set_api_key(api_key)
        if HAS_NEW_GENAI:
            self.client = genai.Client(api_key=api_key)
        elif HAS_LEGACY_GENAI:
            legacy_genai.configure(api_key=api_key)

    def query(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Alias for answer_query."""
        return self.answer_query(query, top_k=top_k)

    def answer_query(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Executes full RAG workflow: Retrieval -> Augmented Prompt -> Generation -> Citations."""
        start_time = time.time()

        # Step 1: Semantic Retrieval
        retrieved_results = self.vector_store.similarity_search(query, top_k=top_k)

        if not retrieved_results:
            return {
                "answer": "No documents are currently indexed. Please upload a PDF or text document first.",
                "sources": [],
                "latency_ms": 0,
                "model": self.model_name
            }

        # Step 2: Build Context & Citations list
        context_blocks = []
        sources = []

        for chunk, score in retrieved_results:
            context_blocks.append(
                f"--- [Source: {chunk.doc_name} | Page: {chunk.page_number} | Relevance: {score:.2f}] ---\n{chunk.text}\n"
            )
            sources.append({
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "relevance_score": round(score, 3),
                "snippet": chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
            })

        full_context = "\n".join(context_blocks)

        user_content = f"""Retrieved Document Context:
{full_context}

User Query:
{query}

Please formulate an evidence-backed answer citing the sources."""

        # Step 3: LLM Generation
        answer_text = ""
        if self.api_key:
            if HAS_NEW_GENAI and self.client:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=RAG_SYSTEM_PROMPT,
                            temperature=0.2,
                        )
                    )
                    answer_text = response.text
                except Exception as e:
                    answer_text = f"Error during Gemini generation: {str(e)}"
            elif HAS_LEGACY_GENAI:
                try:
                    # Model mapping for legacy
                    legacy_name = "gemini-1.5-flash" if "flash" in self.model_name else "gemini-1.5-pro"
                    model = legacy_genai.GenerativeModel(
                        model_name=legacy_name,
                        system_instruction=RAG_SYSTEM_PROMPT
                    )
                    res = model.generate_content(user_content)
                    answer_text = res.text
                except Exception as e:
                    answer_text = f"Error during Legacy Gemini generation: {str(e)}"
            else:
                answer_text = "⚠️ Gemini SDK (`google-genai` or `google-generativeai`) is not installed. Please run `pip install -r requirements.txt`."
        else:
            answer_text = (
                "⚠️ **No Google Gemini API key provided.**\n\n"
                "Top relevant chunks were successfully retrieved from the vector index:\n\n"
                + "\n\n".join([f"> **[{s['doc_name']}, Page {s['page_number']}]**: {s['snippet']}" for s in sources])
            )

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "answer": answer_text,
            "sources": sources,
            "latency_ms": latency_ms,
            "model": self.model_name
        }
