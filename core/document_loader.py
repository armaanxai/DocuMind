import os
import re
from typing import List, Dict, Any, Callable, Optional
from pypdf import PdfReader

class DocumentChunk:
    def __init__(self, text: str, doc_name: str, page_number: int, chunk_id: int):
        self.text = text
        self.doc_name = doc_name
        self.page_number = page_number
        self.chunk_id = chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id
        }

class DocumentLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> List[DocumentChunk]:
        """Auto-detects file type (PDF, TXT, MD) and loads chunks."""
        doc_name = os.path.basename(file_path)
        if file_path.lower().endswith(".pdf"):
            return self.load_pdf(file_path, doc_name=doc_name)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.load_text(content, doc_name=doc_name)

    def load_pdf(
        self,
        file_path_or_buffer,
        doc_name: str = "Uploaded Document",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_pages: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Extracts text from PDF page-by-page and splits into semantic chunks."""
        if hasattr(file_path_or_buffer, "seek"):
            try:
                file_path_or_buffer.seek(0)
            except Exception:
                pass

        reader = PdfReader(file_path_or_buffer)
        all_chunks = []
        global_chunk_id = 0

        total_pages = len(reader.pages)
        limit = min(total_pages, max_pages) if max_pages else total_pages

        for page_idx in range(limit):
            if progress_callback and page_idx % 20 == 0:
                progress_callback(page_idx + 1, limit)

            try:
                page_text = reader.pages[page_idx].extract_text() or ""
            except Exception:
                page_text = ""
                
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            if not page_text:
                continue

            page_chunks = self._chunk_text(page_text)
            for chunk_str in page_chunks:
                all_chunks.append(
                    DocumentChunk(
                        text=chunk_str,
                        doc_name=doc_name,
                        page_number=page_idx + 1,
                        chunk_id=global_chunk_id
                    )
                )
                global_chunk_id += 1

        if progress_callback:
            progress_callback(limit, limit)

        return all_chunks

    def load_text(self, text_content: str, doc_name: str = "Raw Document") -> List[DocumentChunk]:
        """Loads and chunks raw string text or markdown."""
        cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
        if not cleaned_text:
            return []
        chunks = self._chunk_text(cleaned_text)
        return [
            DocumentChunk(
                text=c,
                doc_name=doc_name,
                page_number=1,
                chunk_id=i
            )
            for i, c in enumerate(chunks)
        ]

    def _chunk_text(self, text: str) -> List[str]:
        """Sliding window chunking with configurable overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                split_at = text.rfind('. ', start, end)
                if split_at != -1 and split_at > start + (self.chunk_size // 2):
                    end = split_at + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break

        return chunks
