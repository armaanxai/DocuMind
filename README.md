# 🧠 DocuMind // Neural Document Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-RAG_Pipeline-000000?style=for-the-badge&logo=fastapi&logoColor=00F0FF&borderColor=00F0FF" alt="RAG Architecture" />
  <img src="https://img.shields.io/badge/LLM-Gemini_2.5_Flash-000000?style=for-the-badge&logo=google&logoColor=00F0FF&borderColor=00F0FF" alt="Gemini Flash" />
  <img src="https://img.shields.io/badge/Embeddings-text--embedding--004-000000?style=for-the-badge&logo=python&logoColor=00F0FF&borderColor=00F0FF" alt="Text Embedding" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit_Dark-000000?style=for-the-badge&logo=streamlit&logoColor=00F0FF&borderColor=00F0FF" alt="Streamlit UI" />
  <img src="https://img.shields.io/badge/License-MIT-000000?style=for-the-badge&logo=github&logoColor=00F0FF&borderColor=00F0FF" alt="MIT License" />
</p>

---

## ⚡ Executive Overview

**DocuMind AI** is an enterprise-grade **Retrieval-Augmented Generation (RAG)** platform designed to ingest complex multi-page research papers, technical documentation, and corporate PDFs — enabling grounded, hallucination-free conversational intelligence with verifiable source citations.

Built with **Google Gemini models**, **NumPy Cosine Vector Indexing**, and a **Cyberpunk-inspired Streamlit UI**, DocuMind eliminates LLM hallucinations by enforcing strict evidence retrieval prior to answer synthesis.

---

## 🏛️ Architecture Pipeline

```text
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Multi-page PDF │ ───>  │ Intelligent Chunking   │ ───>  │  Vector Embedding    │
│  & Text Docs    │       │ (Sliding Window + Overlap)│     │ (text-embedding-004) │
└─────────────────┘       └────────────────────────┘       └──────────┬───────────┘
                                                                      │
                                                                      ▼
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│ Grounded Answer │ <───  │ Gemini 2.5 Flash       │ <───  │ Cosine Similarity    │
│ + Page Citation │       │ Context-Injected GenAI │       │ Top-K Vector Search  │
└─────────────────┘       └────────────────────────┘       └──────────────────────┘
```

---

## ✨ Key Technical Capabilities

- **📄 Robust Document Ingestion:** Page-by-page extraction from PDF, TXT, and Markdown files using `pypdf` with clean whitespace normalization.
- **✂️ Semantic Sliding Window Chunking:** Configurable chunk sizing (`750 tokens`) with overlap (`120 tokens`) preserving cross-boundary conceptual integrity.
- **⚡ High-Speed Vector Retrieval:** Normalized in-memory matrix projection calculating exact cosine similarities in sub-millisecond execution times.
- **🎯 Verifiable Source Citations:** Every generated response links directly to its source document, exact page number, and text excerpt snippet.
- **🛡️ Hallucination Guardrails:** Strict system instruction forcing the LLM to refuse ungrounded speculation if documents lack evidence.
- **🖥️ Cyberpunk HUD Interface:** Custom CSS dark-mode dashboard with telemetry latency indicators and interactive citation drawers.

---

## 📂 Repository Structure

```bash
DocuMind-AI/
├── core/
│   ├── __init__.py
│   ├── document_loader.py    # Document parsing & sliding window chunker
│   ├── vector_store.py       # Vector embeddings & cosine similarity index
│   └── rag_pipeline.py       # RAG prompt injection & Gemini generation
├── sample_docs/
│   └── transformer_architecture.txt  # Quick-start demonstration paper
├── app.py                    # Streamlit web application & telemetry HUD
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Production git ignore rules
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Armaan385/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Add your free Gemini API key from [Google AI Studio](https://aistudio.google.com/):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Testing with Included Sample Paper
1. In the sidebar, click **"📑 Load Sample Paper"**.
2. Ask any technical question, e.g.:
   - *"What are the sub-layers inside the Transformer encoder?"*
   - *"What was the BLEU score achieved on the English-to-German task?"*
3. Expand **"📖 Retrieved Evidence & Citations"** to inspect the grounded evidence chunks.

---

## 👨‍💻 Engineering & Authorship

**Armaan Chaudhary (`@Armaan385`)**  
*Data Science, Machine Learning & Neural Systems Undergrad*  

- 🌐 **GitHub:** [@Armaan385](https://github.com/Armaan385)  
- 💼 **LinkedIn:** [Armaan Chaudhary](https://www.linkedin.com/in/armaan-chaudhary-1a5263383/)  
- 📧 **Email:** armaan11chaudhary@gmail.com  

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for details.
