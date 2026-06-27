import streamlit as st
import os
from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #a78bfa !important; }

[data-testid="stChatMessage"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 12px !important; margin-bottom: 10px;
}
[data-testid="stChatInputTextArea"] {
    background: #161b27 !important; color: #e2e8f0 !important;
    border: 1px solid #7c3aed !important; border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 13px !important;
    padding: 8px 16px !important; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #161b27 !important; color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important; border-radius: 8px !important;
}
.streamlit-expanderHeader {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 8px !important; color: #a78bfa !important;
    font-family: 'Space Mono', monospace !important; font-size: 12px !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important; border: 1px solid #1e2330 !important;
    border-radius: 0 0 8px 8px !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #161b27 !important; border: 2px dashed #7c3aed55 !important;
    border-radius: 10px !important; padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7c3aed !important;
}

/* Chips and badges */
.source-chip {
    display: inline-block; background: #1e2330; border: 1px solid #7c3aed44;
    color: #a78bfa; padding: 4px 10px; border-radius: 20px;
    font-size: 11px; font-family: 'Space Mono', monospace; margin: 3px;
}
.badge-success { background:#064e3b;color:#34d399;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-web     { background:#1e3050;color:#60a5fa;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-warn    { background:#3b2000;color:#fbbf24;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-doc     { background:#1a1050;color:#c084fc;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-memory  { background:#0f2030;color:#38bdf8;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-lang    { background:#1a2820;color:#4ade80;padding:3px 10px;border-radius:20px;font-size:11px;font-family:'Space Mono',monospace; }

/* Memory indicator */
.memory-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0f1929; border: 1px solid #1e3a5f;
    color: #7dd3fc; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-family: 'Space Mono', monospace;
}

[data-testid="metric-container"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 10px !important; padding: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #161b27 !important; border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161b27 !important; border-radius: 8px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #94a3b8 !important;
    font-family: 'Space Mono', monospace !important; font-size: 11px !important;
    border-radius: 6px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #7c3aed33 !important; color: #a78bfa !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #7c3aed55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
hr { border-color: #1e2330 !important; }
.stAlert { border-radius: 8px !important; }

/* Doc upload file display */
.doc-file-item {
    display: flex; align-items: center; gap: 8px;
    background: #1a1030; border: 1px solid #7c3aed33;
    border-radius: 8px; padding: 6px 10px; margin: 4px 0;
    font-size: 12px; font-family: 'Space Mono', monospace; color: #c084fc;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "rag": None,
        "messages": [],
        "sources_loaded": [],
        "groq_key_set": False,
        "uploaded_docs": [],   # list of {name, chunks, size}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

LANG_EMOJI = {"hindi": "🇮🇳 Hindi", "hinglish": "🌐 Hinglish", "english": "🇬🇧 English"}
DOC_ICONS  = {"pdf": "📕", "docx": "📘", "doc": "📘", "txt": "📄"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🤖 RAG Chatbot")
    st.markdown("*Retrieval-Augmented Generation*")
    st.divider()

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 API Keys")
    groq_key   = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.text_input(
        "Tavily API Key (Live Web Search)",
        type="password",
        placeholder="tvly-...",
        help="Free at app.tavily.com — enables real-time web search when no docs match",
    )

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        if not st.session_state.groq_key_set:
            st.session_state.groq_key_set = True
            st.session_state.rag = RAGEngine(
                groq_api_key=groq_key,
                tavily_api_key=tavily_key,
            )
        if tavily_key and st.session_state.rag:
            st.session_state.rag.set_tavily_key(tavily_key)

        col1, col2 = st.columns(2)
        col1.markdown('<span class="badge-success">✓ Groq</span>', unsafe_allow_html=True)
        if tavily_key:
            col2.markdown('<span class="badge-web">✓ Web Search</span>', unsafe_allow_html=True)
        else:
            col2.markdown('<span class="badge-warn">⚠ No Web</span>', unsafe_allow_html=True)
    else:
        st.info("⚠️ Add your Groq API key to start")

    st.markdown(
        "<small style='color:#64748b'>Tavily free: 1,000 searches/month — <b>app.tavily.com</b></small>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Groq LLM",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    )
    if st.session_state.rag:
        st.session_state.rag.model = model
    st.divider()

    # ── Knowledge Sources (tabbed) ─────────────────────────────────────────────
    st.markdown("### 📥 Add Knowledge Sources")
    tab_doc, tab_url, tab_text = st.tabs(["📎 Documents", "🌐 URL", "📝 Text"])

    # ── Document Upload Tab ───────────────────────────────────────────────────
    with tab_doc:
        st.markdown(
            "<small style='color:#94a3b8'>Upload PDF, DOCX, or TXT files</small>",
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "doc", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("📂 Index Documents", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not uploaded_files:
                st.error("Select at least one file!")
            else:
                for uf in uploaded_files:
                    # Skip already indexed docs
                    already = any(d["name"] == uf.name for d in st.session_state.uploaded_docs)
                    if already:
                        st.info(f"⏭ Already indexed: {uf.name}")
                        continue

                    with st.spinner(f"Indexing {uf.name}…"):
                        try:
                            file_bytes = uf.read()
                            result = st.session_state.rag.add_document(file_bytes, uf.name)
                            size_kb = len(file_bytes) / 1024
                            st.session_state.uploaded_docs.append({
                                "name": uf.name,
                                "chunks": result["chunks"],
                                "size_kb": round(size_kb, 1),
                            })
                            st.session_state.sources_loaded.append({
                                "type": "doc",
                                "src": uf.name,
                            })
                            st.success(
                                f"✓ **{uf.name}** — {result['chunks']} chunks, "
                                f"{size_kb:.1f} KB"
                            )
                        except Exception as e:
                            st.error(f"❌ {uf.name}: {e}")

        # Show indexed docs
        if st.session_state.uploaded_docs:
            st.markdown("<br>", unsafe_allow_html=True)
            for doc in st.session_state.uploaded_docs:
                ext = doc["name"].rsplit(".", 1)[-1].lower()
                icon = DOC_ICONS.get(ext, "📄")
                short = doc["name"][:28] + "…" if len(doc["name"]) > 28 else doc["name"]
                st.markdown(
                    f'<div class="doc-file-item">'
                    f'{icon} <span style="flex:1">{short}</span>'
                    f'<span style="color:#64748b;font-size:10px">{doc["chunks"]}c · {doc["size_kb"]}KB</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── URL Tab ───────────────────────────────────────────────────────────────
    with tab_url:
        url_input   = st.text_input("Website URL", placeholder="https://example.com")
        crawl_depth = st.slider("Crawl depth", 1, 3, 1)
        if st.button("🕷️ Scrape & Index", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not url_input:
                st.error("Enter a URL!")
            else:
                with st.spinner(f"Scraping {url_input}..."):
                    try:
                        result = st.session_state.rag.add_url(url_input, depth=crawl_depth)
                        st.session_state.sources_loaded.append({"type": "url", "src": url_input})
                        st.success(f"✓ {result['chunks']} chunks from {result['pages']} page(s)")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Text Tab ──────────────────────────────────────────────────────────────
    with tab_text:
        raw_text   = st.text_area("Paste text / docs", height=130, placeholder="Paste any text…")
        text_label = st.text_input("Label (optional)", placeholder="My Document")
        if st.button("📄 Index Text", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not raw_text.strip():
                st.error("Enter some text!")
            else:
                with st.spinner("Indexing..."):
                    try:
                        result = st.session_state.rag.add_text(raw_text, label=text_label or "Manual Text")
                        st.session_state.sources_loaded.append({"type": "text", "src": text_label or "Manual Text"})
                        st.success(f"✓ {result['chunks']} chunks indexed")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Loaded Sources ─────────────────────────────────────────────────────────
    non_doc_sources = [s for s in st.session_state.sources_loaded if s["type"] != "doc"]
    if non_doc_sources:
        st.divider()
        st.markdown("### 📚 Other Sources")
        for s in non_doc_sources:
            icon  = "🌐" if s["type"] == "url" else "📝"
            label = s["src"][:35] + "…" if len(s["src"]) > 35 else s["src"]
            st.markdown(f'<span class="source-chip">{icon} {label}</span>', unsafe_allow_html=True)

    st.divider()

    # ── RAG Settings ──────────────────────────────────────────────────────────
    st.markdown("### 🎛️ RAG Settings")
    top_k       = st.slider("Top-K chunks", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    if st.session_state.rag:
        st.session_state.rag.top_k       = top_k
        st.session_state.rag.temperature = temperature

    # ── Index Stats ──────────────────────────────────────────────────────────
    if st.session_state.rag and st.session_state.rag.vector_store.total > 0:
        st.divider()
        st.markdown("### 📊 Stats")
        stats = st.session_state.rag.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", stats["chunks"])
        c2.metric("Sources", stats["sources"])
        c3.metric("Turns", stats["memory_turns"])

    st.divider()

    # ── Actions ───────────────────────────────────────────────────────────────
    col_mem, col_all = st.columns(2)
    with col_mem:
        if st.button("🧠 Clear Memory", use_container_width=True):
            if st.session_state.rag:
                st.session_state.rag.clear_memory()
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.rerun()
    with col_all:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.sources_loaded = []
            st.session_state.uploaded_docs  = []
            if st.session_state.rag:
                st.session_state.rag.clear()
            st.success("Cleared!")
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("# 💬 RAG Chatbot")

# Status bar
has_docs  = bool(st.session_state.sources_loaded)
has_web   = st.session_state.rag and st.session_state.rag.live_search
mem_turns = st.session_state.rag.get_stats()["memory_turns"] if st.session_state.rag else 0

status_parts = []
if has_docs:
    doc_count = len(st.session_state.uploaded_docs)
    url_count = sum(1 for s in st.session_state.sources_loaded if s["type"] == "url")
    txt_count = sum(1 for s in st.session_state.sources_loaded if s["type"] == "text")
    label_parts = []
    if doc_count: label_parts.append(f"{doc_count} doc{'s' if doc_count>1 else ''}")
    if url_count: label_parts.append(f"{url_count} URL{'s' if url_count>1 else ''}")
    if txt_count: label_parts.append(f"{txt_count} text")
    status_parts.append(f'<span class="badge-doc">📎 {", ".join(label_parts)}</span>')
else:
    status_parts.append('<span class="badge-warn">⚠ No docs loaded</span>')

if has_web:
    status_parts.append('<span class="badge-web">🌐 Web Search ON</span>')

if mem_turns > 0:
    status_parts.append(f'<span class="badge-memory">🧠 {mem_turns} turn{"s" if mem_turns>1 else ""} in memory</span>')

status_parts.append('<span class="badge-lang">🌏 Hindi / Hinglish / English</span>')

st.markdown(" &nbsp; ".join(status_parts), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Badges on assistant messages
        if msg["role"] == "assistant":
            badges = []
            if msg.get("used_web_search"):
                badges.append('<span class="badge-web">🌐 Live web search</span>')
            if msg.get("language_detected") and msg["language_detected"] != "english":
                lang_label = LANG_EMOJI.get(msg["language_detected"], "")
                badges.append(f'<span class="badge-lang">{lang_label}</span>')
            if badges:
                st.markdown(" ".join(badges), unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
placeholder_text = "Ask in English, Hindi, या Hinglish… | Upload docs in sidebar for doc-based answers"

if prompt := st.chat_input(placeholder_text):
    if not st.session_state.rag:
        st.error("Please add your Groq API key in the sidebar first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.rag.query(prompt)
                answer   = result["answer"]
                sources  = result.get("sources", [])
                used_web = result.get("used_web_search", False)
                lang     = result.get("language_detected", "english")

                st.markdown(answer)

                badges = []
                if used_web:
                    badges.append('<span class="badge-web">🌐 Live web search</span>')
                if lang != "english":
                    lang_label = LANG_EMOJI.get(lang, "")
                    badges.append(f'<span class="badge-lang">{lang_label}</span>')
                if badges:
                    st.markdown(" ".join(badges), unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for src in sources:
                            st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "used_web_search": used_web,
                    "language_detected": lang,
                })

                # Update memory turn display without full rerun
                st.rerun()

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
