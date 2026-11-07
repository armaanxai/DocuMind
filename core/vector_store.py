import numpy as np
from typing import List, Dict, Any, Tuple
from .document_loader import DocumentChunk

# Safe import for Gemini SDKs
try:
    from google import genai
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

class VectorStore:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.set_api_key(api_key)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        if HAS_NEW_GENAI:
            self.client = genai.Client(api_key=api_key)
        elif HAS_LEGACY_GENAI:
            legacy_genai.configure(api_key=api_key)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Fetch embedding from Gemini text-embedding-004 or compute fallback."""
        if self.api_key:
            if HAS_NEW_GENAI and self.client:
                try:
                    response = self.client.models.embed_content(
                        model="text-embedding-004",
                        contents=text
                    )
                    emb = np.array(response.embeddings[0].values, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
                except Exception as e:
                    print(f"[VectorStore] New GenAI embedding failed: {e}. Falling back.")
            elif HAS_LEGACY_GENAI:
                try:
                    res = legacy_genai.embed_content(
                        model="models/text-embedding-004",
                        content=text
                    )
                    emb = np.array(res['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
                except Exception as e:
                    print(f"[VectorStore] Legacy GenAI embedding failed: {e}. Falling back.")

        # Deterministic semantic projection fallback
        return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str, dim: int = 768) -> np.ndarray:
        """Lightweight bag-of-words / character n-gram projection for offline testing."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            idx = abs(hash(w)) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add_documents(self, chunks: List[DocumentChunk]):
        """Embeds and indexes a list of document chunks."""
        if not chunks:
            return

        new_embeddings = []
        for chunk in chunks:
            emb = self._get_embedding(chunk.text)
            new_embeddings.append(emb)

        new_matrix = np.vstack(new_embeddings)

        if self.embeddings is None:
            self.embeddings = new_matrix
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, new_matrix])
            self.chunks.extend(chunks)

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Alias for add_documents."""
        return self.add_documents(chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """Computes cosine similarity of query against all stored chunks and returns top_k."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_emb = self._get_embedding(query)
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, query_emb)
        
        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def clear(self):
        """Clears the current vector index."""
        self.chunks = []
        self.embeddings = None
