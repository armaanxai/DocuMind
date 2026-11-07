import os
import streamlit as st
from dotenv import load_dotenv

from core.document_loader import DocumentLoader
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline

SHARED_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Base Elements */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Ambient Dynamic Glowing Mesh Background */
    .stApp {
        background-color: #070913 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.24) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.22) 0px, transparent 50%),
            radial-gradient(at 50% 50%, rgba(14, 165, 233, 0.14) 0px, transparent 55%),
            radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.18) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(59, 130, 246, 0.16) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        background-size: cover !important;
        color: #f8fafc;
    }

    /* Glassmorphic Translucent Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.68) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
        box-shadow: 10px 0 35px -5px rgba(0, 0, 0, 0.5) !important;
    }

    /* Glass Hero Banner */
    .hero-banner {
        padding: 26px 32px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.58) 100%);
        backdrop-filter: blur(24px) saturate(190%);
        -webkit-backdrop-filter: blur(24px) saturate(190%);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 20px;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.45), 
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.25);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #38bdf8, #ec4899);
        box-shadow: 0 0 14px rgba(168, 85, 247, 0.65);
    }

    /* Glass Pill Tag */
    .pill-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        background: rgba(99, 102, 241, 0.18);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(165, 180, 252, 0.35);
        border-radius: 9999px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #c7d2fe;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
    }

    .brand-heading {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 40%, #c084fc 80%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-shadow: 0 0 30px rgba(192, 132, 252, 0.25);
    }

    .brand-tagline {
        color: #cbd5e1;
        font-size: 0.98rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Glass Stats Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.045);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.25),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.15);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(165, 180, 252, 0.4);
        box-shadow: 0 14px 30px -8px rgba(99, 102, 241, 0.35),
                    inset 0 1px 2px 0 rgba(255, 255, 255, 0.25);
    }

    .stat-number {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }

    .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Dashboard Executive KPI Grid */
    .dashboard-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(18, 24, 42, 0.6) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(165, 180, 252, 0.45);
        box-shadow: 0 16px 36px -8px rgba(99, 102, 241, 0.35);
    }
    .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .kpi-icon {
        font-size: 1.5rem;
    }
    .kpi-badge {
        font-size: 0.70rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(165, 180, 252, 0.3);
        color: #c7d2fe;
        letter-spacing: 0.04em;
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .kpi-lbl {
        font-size: 0.80rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }
    .kpi-sub {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* Dashboard Section Container */
    .dash-section {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 20px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28), inset 0 1px 1px rgba(255, 255, 255, 0.12);
    }

    /* Document inventory rows */
    .inventory-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: background 0.2s ease;
    }
    .inventory-item:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    .inventory-title {
        font-size: 0.90rem;
        font-weight: 600;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .inventory-meta {
        font-size: 0.78rem;
        color: #94a3b8;
        display: flex;
        gap: 14px;
    }

    /* Glass Feature Cards (Homepage Launchpad & Sections) */
    .feature-card {
        background: rgba(255, 255, 255, 0.042);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.14);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(165, 180, 252, 0.45);
        box-shadow: 0 18px 40px -8px rgba(99, 102, 241, 0.35),
                    inset 0 1px 2px 0 rgba(255, 255, 255, 0.25);
    }

    /* Glass Citation Container */
    .citation-container {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-left: 4px solid #818cf8;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
    }

    .citation-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: #a5b4fc;
        margin-bottom: 6px;
    }

    /* Glass Action Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.45) 0%, rgba(168, 85, 247, 0.45) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        box-shadow: 0 6px 20px -4px rgba(0, 0, 0, 0.35),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.75) 0%, rgba(168, 85, 247, 0.75) 100%) !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
        box-shadow: 0 10px 28px -4px rgba(99, 102, 241, 0.55),
                    inset 0 1px 2px 0 rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
    }

    /* Glass Download Button */
    .stDownloadButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.4) 0%, rgba(14, 165, 233, 0.4) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px -4px rgba(16, 185, 129, 0.35),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.25) !important;
        transition: all 0.22s ease !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.7) 0%, rgba(14, 165, 233, 0.7) 100%) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 10px 28px -4px rgba(16, 185, 129, 0.55) !important;
        transform: translateY(-2px) !important;
    }

    /* Glass Form Inputs (Text inputs, Textareas, Selectboxes) */
    div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] {
        background: rgba(18, 24, 42, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 18px rgba(129, 140, 248, 0.4), inset 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }

    /* Glass Chat Input Bar */
    div[data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
    }

    /* Glass Chat Message Bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25),
                    inset 0 1px 1px rgba(255, 255, 255, 0.12) !important;
        margin-bottom: 12px !important;
    }

    /* Glass Expanders */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.035) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.11) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        overflow: hidden;
    }

    /* Glass File Uploader & Dropzone */
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="stFileUploader"] label {
        color: #a5b4fc !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(99, 102, 241, 0.10) 100%) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1.5px dashed rgba(165, 180, 252, 0.45) !important;
        border-radius: 16px !important;
        padding: 20px 14px !important;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
        transition: all 0.25s ease !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #818cf8 !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(99, 102, 241, 0.18) 100%) !important;
        box-shadow: 0 12px 30px -4px rgba(99, 102, 241, 0.4), inset 0 1px 2px rgba(255, 255, 255, 0.25) !important;
    }
    div[data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 10px !important;
        padding: 7px 18px !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.5) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFileUploadDropzone"] button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #a855f7 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.65) !important;
    }
    div[data-testid="stFileUploadDropzone"] small,
    div[data-testid="stFileUploadDropzone"] span {
        color: #cbd5e1 !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stFileUploadDropzone"] svg {
        fill: #a5b4fc !important;
        color: #a5b4fc !important;
    }

    /* Glass Radio Buttons (Indexing Range & Choice Pickers) */
    div[data-testid="stRadio"] label:first-child {
        color: #a5b4fc !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background: rgba(18, 24, 42, 0.6) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        padding: 8px 10px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(99, 102, 241, 0.14) !important;
        border-color: rgba(165, 180, 252, 0.35) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] span {
        color: #f1f5f9 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    /* Radio Dot Glow */
    div[data-testid="stRadio"] input[type="radio"]:checked + div {
        border-color: #818cf8 !important;
        background-color: transparent !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:checked + div > div {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        box-shadow: 0 0 12px rgba(129, 140, 248, 0.9) !important;
    }

    /* Sidebar Glass Header Card */
    .sidebar-header-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(99, 102, 241, 0.10) 100%);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.18);
    }
    .sidebar-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(165, 180, 252, 0.3);
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #c7d2fe;
        margin-bottom: 8px;
    }
    .sidebar-header-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .sidebar-header-subtitle {
        font-size: 0.80rem;
        color: #94a3b8;
        margin-top: 4px;
        line-height: 1.45;
    }

    /* Sidebar Active Documents Section */
    .sidebar-stat-chip {
        display: inline-flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 7px 12px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    .sidebar-stat-chip b {
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Sleek Translucent Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(10, 14, 26, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(165, 180, 252, 0.45);
    }
</style>
"""

def init_app():
    """Initializes environment, state, styles, and enforces API key access gate across all pages."""
    load_dotenv()
    st.markdown(SHARED_CSS, unsafe_allow_html=True)

    # Initialize Session States
    if "api_key" not in st.session_state:
        st.session_state.api_key = None

    if "api_provider" not in st.session_state:
        st.session_state.api_provider = "auto"

    if "model_name" not in st.session_state:
        st.session_state.model_name = None

    if "base_url" not in st.session_state:
        st.session_state.base_url = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(api_key=st.session_state.api_key)

    if "loader" not in st.session_state:
        st.session_state.loader = DocumentLoader(chunk_size=750, chunk_overlap=120)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = []

    if "status_banner" not in st.session_state:
        st.session_state.status_banner = None

    if "interactive_quiz" not in st.session_state:
        st.session_state.interactive_quiz = None

    if "user_quiz_answers" not in st.session_state:
        st.session_state.user_quiz_answers = {}

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = None

    if "suggested_prompt" not in st.session_state:
        st.session_state.suggested_prompt = None

    if "active_exam_paper" not in st.session_state:
        st.session_state.active_exam_paper = None

    if "active_dossier" not in st.session_state:
        st.session_state.active_dossier = None

    if "active_mind_map" not in st.session_state:
        st.session_state.active_mind_map = None

    if "active_summary" not in st.session_state:
        st.session_state.active_summary = None

    if "active_comparison" not in st.session_state:
        st.session_state.active_comparison = None

    if "anki_data" not in st.session_state:
        st.session_state.anki_data = None

    if "exported_notes_text" not in st.session_state:
        st.session_state.exported_notes_text = None

    if "exported_notes_html" not in st.session_state:
        st.session_state.exported_notes_html = None

    # STRICT ACCESS GATE: Website refuses to run without an active API Key
    if not st.session_state.get("api_key"):
        st.markdown("""
        <div class="hero-banner" style="text-align: center; margin-top: 24px;">
            <div class="pill-tag">🔐 SECURITY GATEWAY // API KEY REQUIRED</div>
            <div class="brand-heading" style="font-size: 2.3rem;">DocuMind Access Gate</div>
            <div class="brand-tagline" style="max-width: 660px; margin: 8px auto 0 auto;">
                DocuMind requires an active LLM API Key to run. Enter any API key below — 
                the platform supports <b>Groq (Free/Fast)</b>, <b>OpenAI</b>, <b>OpenRouter</b>, <b>DeepSeek</b>, <b>Anthropic Claude</b>, or <b>Custom / Gemini</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        gate_c1, gate_c2, gate_c3 = st.columns([1, 2, 1])
        with gate_c2:
            st.markdown("""
            <div class="dash-section" style="padding: 24px 28px;">
                <div style="font-size: 1.10rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                    🔑 Universal API Key Activation
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 16px;">
                    Choose your provider or let DocuMind auto-detect your key format automatically.
                </div>
            """, unsafe_allow_html=True)

            provider_choice = st.selectbox(
                "LLM Provider:",
                options=[
                    "⚡ Auto-Detect (Paste Any Key)",
                    "Groq (Llama 3.3 70B - Ultra Fast)",
                    "OpenAI (GPT-4o Mini / 4o)",
                    "OpenRouter (All LLMs / Multi-model)",
                    "DeepSeek (DeepSeek V3 / Chat)",
                    "Anthropic Claude (3.5 Sonnet)",
                    "Custom / Local Endpoint (Ollama, vLLM)",
                    "Google Gemini"
                ],
                key="gate_provider_select"
            )

            env_key_hint = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
            key_input = st.text_input(
                "Enter API Key:",
                value="",
                type="password",
                placeholder="Paste your gsk_... or sk-... key here",
                key="gate_key_input",
                help="Your API key remains strictly in memory for this session and is never uploaded anywhere."
            )

            custom_base_url = None
            custom_model = None
            if "Custom" in provider_choice:
                custom_base_url = st.text_input("Custom Base URL:", value="http://localhost:11434/v1", key="gate_base_url")
                custom_model = st.text_input("Custom Model Name:", value="llama3", key="gate_model_name")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🚀 Unlock & Activate DocuMind", use_container_width=True, key="btn_unlock_gate"):
                raw_key = key_input.strip()
                if not raw_key and env_key_hint:
                    raw_key = env_key_hint.strip()

                if raw_key and len(raw_key) >= 6:
                    info = RAGPipeline.detect_provider(raw_key, user_provider=provider_choice)
                    st.session_state.api_key = raw_key
                    st.session_state.api_provider = info["provider"]
                    st.session_state.model_name = custom_model or info["model"]
                    st.session_state.base_url = custom_base_url or info["base_url"]
                    st.session_state.status_banner = ("success", f"DocuMind unlocked with {st.session_state.api_provider.upper()} ({st.session_state.model_name})!")
                    st.rerun()
                else:
                    st.error("⚠️ Please enter a valid API key to activate DocuMind.")

            st.markdown("""
                <div style="font-size: 0.76rem; color: #64748b; margin-top: 14px; text-align: center; line-height: 1.5;">
                    💡 Need a free key? Get a free high-speed API key from <b><a href="https://console.groq.com" target="_blank" style="color: #818cf8;">console.groq.com</a></b> or <b><a href="https://openrouter.ai" target="_blank" style="color: #818cf8;">openrouter.ai</a></b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.stop()

    # Active Pipeline Sync
    st.session_state.vector_store.set_api_key(st.session_state.api_key)
    pipeline = RAGPipeline(
        vector_store=st.session_state.vector_store,
        api_key=st.session_state.api_key,
        provider=st.session_state.api_provider,
        model_name=st.session_state.model_name,
        base_url=st.session_state.base_url
    )
    return pipeline

def render_sidebar():
    """Renders the document management sidebar consistently across all pages."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header-card">
            <div class="sidebar-header-badge">⚡ KNOWLEDGE ENGINE</div>
            <div class="sidebar-header-title">Upload Books & PDFs</div>
            <div class="sidebar-header-subtitle">
                Support for 1,000+ page books & notes with instant lexical + neural indexing.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Choose PDF or Text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="global_file_uploader"
        )

        index_scope = st.radio(
            "Indexing Range:",
            options=["Full Book (All Pages)", "Quick Index (First 200 Pages)"],
            index=0,
            key="global_index_scope"
        )
        max_pages_limit = 200 if "Quick" in index_scope else None

        progress_placeholder = st.empty()
        status_text_placeholder = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Index Files", use_container_width=True, key="global_btn_index"):
                if uploaded_files:
                    new_chunks = []
                    skipped_scanned = []
                    for f in uploaded_files:
                        try:
                            if hasattr(f, "seek"):
                                f.seek(0)
                            if f.name.lower().endswith(".pdf"):
                                def on_progress(current, total):
                                    pct = int((current / max(total, 1)) * 100)
                                    progress_placeholder.progress(pct)
                                    status_text_placeholder.caption(f"📖 Processing `{f.name}`: Page {current}/{total} ({pct}%)...")

                                chunks = st.session_state.loader.load_pdf(
                                    f,
                                    doc_name=f.name,
                                    progress_callback=on_progress,
                                    max_pages=max_pages_limit
                                )
                            else:
                                text = f.read().decode("utf-8", errors="ignore")
                                chunks = st.session_state.loader.load_text(text, doc_name=f.name)
                            
                            if chunks:
                                new_chunks.extend(chunks)
                                if f.name not in st.session_state.indexed_docs:
                                    st.session_state.indexed_docs.append(f.name)
                            else:
                                skipped_scanned.append(f.name)
                        except Exception as e:
                            st.error(f"Error reading {f.name}: {e}")

                    if new_chunks:
                        status_text_placeholder.caption("⚡ Building search index...")
                        st.session_state.vector_store.set_api_key(st.session_state.api_key)
                        st.session_state.vector_store.add_documents(new_chunks)
                        progress_placeholder.empty()
                        status_text_placeholder.empty()
                        st.session_state.status_banner = ("success", f"Indexed {len(new_chunks)} passages across {len(st.session_state.indexed_docs)} file(s)!")
                        st.rerun()
                    elif skipped_scanned:
                        progress_placeholder.empty()
                        status_text_placeholder.empty()
                        st.session_state.status_banner = ("warning", f"No readable text in {', '.join(skipped_scanned)}.")
                        st.rerun()
                else:
                    st.warning("Please choose a file first.")

        with col2:
            if st.button("📑 Load Sample", use_container_width=True, key="global_btn_sample"):
                sample_path = os.path.join("sample_docs", "transformer_architecture.txt")
                if os.path.exists(sample_path):
                    with st.spinner("Loading sample paper..."):
                        with open(sample_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        chunks = st.session_state.loader.load_text(content, doc_name="Transformer_Architecture_Paper.txt")
                        st.session_state.vector_store.set_api_key(st.session_state.api_key)
                        st.session_state.vector_store.add_documents(chunks)
                        if "Transformer_Architecture_Paper.txt" not in st.session_state.indexed_docs:
                            st.session_state.indexed_docs.append("Transformer_Architecture_Paper.txt")
                        st.session_state.status_banner = ("success", f"Loaded sample document ({len(chunks)} passages ready).")
                        st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.82rem; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
            📊 Active Knowledge Base
        </div>
        """, unsafe_allow_html=True)
        total_chunks = len(st.session_state.vector_store.chunks)
        active_docs = list(set(st.session_state.indexed_docs))
        
        st.markdown(f"""
        <div class="sidebar-stat-chip">
            <span>📚 Loaded Files</span>
            <b>{len(active_docs)}</b>
        </div>
        <div class="sidebar-stat-chip">
            <span>🧩 Search Passages</span>
            <b>{total_chunks}</b>
        </div>
        """, unsafe_allow_html=True)
        if active_docs:
            for doc in active_docs:
                st.caption(f"• 📖 {doc}")

        if st.button("🗑️ Clear Library", use_container_width=True, key="global_btn_clear"):
            st.session_state.vector_store.clear()
            st.session_state.indexed_docs = []
            st.session_state.chat_history = []
            st.session_state.interactive_quiz = None
            st.session_state.user_quiz_answers = {}
            st.session_state.quiz_score = None
            st.session_state.active_exam_paper = None
            st.session_state.active_dossier = None
            st.session_state.active_mind_map = None
            st.session_state.active_summary = None
            st.session_state.active_comparison = None
            st.session_state.anki_data = None
            st.session_state.exported_notes_text = None
            st.session_state.exported_notes_html = None
            st.session_state.status_banner = ("info", "Document library cleared.")
            st.rerun()

        # Active Intelligence Engine Status Card
        active_prov = st.session_state.get("api_provider", "Universal").upper()
        active_model = st.session_state.get("model_name", "AI Engine")
        raw_k = st.session_state.get("api_key", "")
        masked_k = (raw_k[:4] + "••••" + raw_k[-4:]) if len(raw_k) > 8 else "••••••••"

        st.markdown(f"""
        <div class="sidebar-header-card" style="margin-top: 14px; background: rgba(99, 102, 241, 0.08); border-color: rgba(165, 180, 252, 0.25);">
            <div class="sidebar-header-badge">🟢 ENGINE ONLINE</div>
            <div style="font-size: 1.0rem; font-weight: 800; color: #ffffff;">{active_prov}</div>
            <div style="font-size: 0.76rem; color: #a5b4fc; margin-top: 2px;">{active_model}</div>
            <div style="font-size: 0.72rem; color: #64748b; font-family: monospace; margin-top: 4px;">{masked_k}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔒 Lock / Switch API Key", use_container_width=True, key="global_btn_lock"):
            st.session_state.api_key = None
            st.rerun()

        with st.expander("⚙️ Key & Model Settings", expanded=False):
            updated_key = st.text_input("Active API Key", value=st.session_state.api_key, type="password", key="global_key_input")
            if updated_key and updated_key != st.session_state.api_key:
                info = RAGPipeline.detect_provider(updated_key)
                st.session_state.api_key = updated_key
                st.session_state.api_provider = info["provider"]
                st.session_state.model_name = info["model"]
                st.session_state.base_url = info["base_url"]
                st.session_state.vector_store.set_api_key(updated_key)
                st.rerun()
