import streamlit as st
import os
import streamlit.components.v1 as components
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
[data-testid="stFileUploader"] {
    background: #161b27 !important; border: 2px dashed #7c3aed55 !important;
    border-radius: 10px !important; padding: 8px !important;
}
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
[data-testid="metric-container"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 10px !important; padding: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #161b27 !important; border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161b27 !important; border-radius: 8px !important; gap: 4px !important;
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
.doc-file-item {
    display: flex; align-items: center; gap: 8px;
    background: #1a1030; border: 1px solid #7c3aed33;
    border-radius: 8px; padding: 6px 10px; margin: 4px 0;
    font-size: 12px; font-family: 'Space Mono', monospace; color: #c084fc;
}
/* Give room at bottom for the custom bar iframe */
.main .block-container { padding-bottom: 130px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "rag": None,
        "messages": [],
        "sources_loaded": [],
        "groq_key_set": False,
        "uploaded_docs": [],
        "inline_upload_key": 0,
        # FIX: voice_transcript holds text injected from the mic widget
        "voice_transcript": "",
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

    st.markdown("### 🔑 API Keys")
    groq_key   = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.text_input("Tavily API Key (Live Web Search)", type="password",
                               placeholder="tvly-...",
                               help="Free at app.tavily.com")

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        if not st.session_state.groq_key_set:
            st.session_state.groq_key_set = True
            st.session_state.rag = RAGEngine(groq_api_key=groq_key, tavily_api_key=tavily_key)
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
    st.markdown("<small style='color:#64748b'>Tavily free: 1,000 searches/month — <b>app.tavily.com</b></small>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Model")
    model = st.selectbox("Groq LLM",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"])
    if st.session_state.rag:
        st.session_state.rag.model = model
    st.divider()

    st.markdown("### 📥 Add Knowledge Sources")
    tab_doc, tab_url, tab_text = st.tabs(["📎 Documents", "🌐 URL", "📝 Text"])

    with tab_doc:
        st.markdown("<small style='color:#94a3b8'>Upload PDF, DOCX, or TXT</small>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Drop files", type=["pdf","docx","doc","txt"],
                                          accept_multiple_files=True, label_visibility="collapsed")
        if st.button("📂 Index Documents", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not uploaded_files:
                st.error("Select at least one file!")
            else:
                for uf in uploaded_files:
                    already = any(d["name"] == uf.name for d in st.session_state.uploaded_docs)
                    if already:
                        st.info(f"⏭ Already indexed: {uf.name}")
                        continue
                    with st.spinner(f"Indexing {uf.name}…"):
                        try:
                            file_bytes = uf.read()
                            result = st.session_state.rag.add_document(file_bytes, uf.name)
                            size_kb = len(file_bytes) / 1024
                            st.session_state.uploaded_docs.append(
                                {"name": uf.name, "chunks": result["chunks"], "size_kb": round(size_kb,1)})
                            st.session_state.sources_loaded.append({"type":"doc","src":uf.name})
                            st.success(f"✓ **{uf.name}** — {result['chunks']} chunks")
                        except Exception as e:
                            st.error(f"❌ {uf.name}: {e}")
        if st.session_state.uploaded_docs:
            st.markdown("<br>", unsafe_allow_html=True)
            for doc in st.session_state.uploaded_docs:
                ext  = doc["name"].rsplit(".",1)[-1].lower()
                icon = DOC_ICONS.get(ext, "📄")
                short = doc["name"][:28]+"…" if len(doc["name"])>28 else doc["name"]
                st.markdown(
                    f'<div class="doc-file-item">{icon} <span style="flex:1">{short}</span>'
                    f'<span style="color:#64748b;font-size:10px">{doc["chunks"]}c · {doc["size_kb"]}KB</span></div>',
                    unsafe_allow_html=True)

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
                        st.session_state.sources_loaded.append({"type":"url","src":url_input})
                        st.success(f"✓ {result['chunks']} chunks from {result['pages']} page(s)")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_text:
        raw_text   = st.text_area("Paste text", height=130, placeholder="Paste any text…")
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
                        st.session_state.sources_loaded.append({"type":"text","src":text_label or "Manual Text"})
                        st.success(f"✓ {result['chunks']} chunks indexed")
                    except Exception as e:
                        st.error(f"Error: {e}")

    non_doc_sources = [s for s in st.session_state.sources_loaded if s["type"] != "doc"]
    if non_doc_sources:
        st.divider()
        st.markdown("### 📚 Other Sources")
        for s in non_doc_sources:
            icon  = "🌐" if s["type"]=="url" else "📝"
            label = s["src"][:35]+"…" if len(s["src"])>35 else s["src"]
            st.markdown(f'<span class="source-chip">{icon} {label}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎛️ RAG Settings")
    top_k       = st.slider("Top-K chunks", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    if st.session_state.rag:
        st.session_state.rag.top_k       = top_k
        st.session_state.rag.temperature = temperature

    if st.session_state.rag and st.session_state.rag.vector_store.total > 0:
        st.divider()
        st.markdown("### 📊 Stats")
        stats = st.session_state.rag.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", stats["chunks"])
        c2.metric("Sources", stats["sources"])
        c3.metric("Turns", stats["memory_turns"])

    st.divider()
    col_mem, col_all = st.columns(2)
    with col_mem:
        if st.button("🧠 Clear Memory", use_container_width=True):
            if st.session_state.rag: st.session_state.rag.clear_memory()
            st.session_state.messages = []
            st.rerun()
    with col_all:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.sources_loaded = []
            st.session_state.uploaded_docs  = []
            if st.session_state.rag: st.session_state.rag.clear()
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("# 💬 RAG Chatbot")

has_docs  = bool(st.session_state.sources_loaded)
has_web   = st.session_state.rag and st.session_state.rag.live_search
mem_turns = st.session_state.rag.get_stats()["memory_turns"] if st.session_state.rag else 0

status_parts = []
if has_docs:
    doc_count = len(st.session_state.uploaded_docs)
    url_count = sum(1 for s in st.session_state.sources_loaded if s["type"]=="url")
    txt_count = sum(1 for s in st.session_state.sources_loaded if s["type"]=="text")
    parts = []
    if doc_count: parts.append(f"{doc_count} doc{'s' if doc_count>1 else ''}")
    if url_count: parts.append(f"{url_count} URL{'s' if url_count>1 else ''}")
    if txt_count: parts.append(f"{txt_count} text")
    status_parts.append(f'<span class="badge-doc">📎 {", ".join(parts)}</span>')
else:
    status_parts.append('<span class="badge-warn">⚠ No docs loaded</span>')
if has_web:
    status_parts.append('<span class="badge-web">🌐 Web Search ON</span>')
if mem_turns > 0:
    status_parts.append(f'<span class="badge-memory">🧠 {mem_turns} turn{"s" if mem_turns>1 else ""} in memory</span>')
status_parts.append('<span class="badge-lang">🌏 Hindi / Hinglish / English</span>')
st.markdown(" &nbsp; ".join(status_parts), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        padding: 60px 20px 40px 20px; text-align:center;
    ">
        <div style="
            font-size: 52px; margin-bottom: 16px;
            filter: drop-shadow(0 0 20px #7c3aed88);
        ">🤖</div>
        <h2 style="
            font-family: 'Space Mono', monospace;
            color: #a78bfa; font-size: 26px;
            margin-bottom: 10px; font-weight: 700;
        ">Hi! Welcome 👋</h2>
        <p style="
            color: #94a3b8; font-size: 16px; max-width: 480px;
            line-height: 1.6; margin-bottom: 24px;
        ">How may I help you today?<br>
        Ask me anything in <b style='color:#c084fc'>English</b>,
        <b style='color:#c084fc'>Hindi</b>, or
        <b style='color:#c084fc'>Hinglish</b> —
        or upload a document to chat with it.</p>
        <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:center;">
            <span style="background:#1e2330;border:1px solid #7c3aed44;color:#a78bfa;
                padding:6px 14px;border-radius:20px;font-size:13px;">
                💬 Ask a question
            </span>
            <span style="background:#1e2330;border:1px solid #7c3aed44;color:#a78bfa;
                padding:6px 14px;border-radius:20px;font-size:13px;">
                📎 Upload a document
            </span>
            <span style="background:#1e2330;border:1px solid #7c3aed44;color:#a78bfa;
                padding:6px 14px;border-radius:20px;font-size:13px;">
                🎤 Speak your query
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # No badges or sources shown — keep chat clean


# ── Single clean input: st.chat_input (typing) ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Input area: hidden helpers + floating buttons (+ and mic) over chat_input bar
# ─────────────────────────────────────────────────────────────────────────────

# 1. Hide helper widgets off-screen but keep in DOM so Streamlit can read them
st.markdown("""
<style>
/* Hide voice hidden input */
div[data-testid="stTextInput"]:has(input[aria-label="voice_hidden"]) {
    position: fixed !important; bottom: -400px !important;
    opacity: 0 !important; pointer-events: none !important; z-index: -1 !important;
}
/* Hide the inline file uploader widget */
div[data-testid="stFileUploader"]:has(input[aria-label="chat_attach"]) {
    position: fixed !important; bottom: -400px !important;
    opacity: 0 !important; pointer-events: none !important; z-index: -1 !important;
}
.main .block-container { padding-bottom: 90px !important; }
</style>
""", unsafe_allow_html=True)

# 2. Hidden voice text_input — mic JS populates this, Streamlit reads it
voice_text = st.text_input("voice_hidden", key="voice_input_field",
                            label_visibility="hidden")
if voice_text and voice_text != st.session_state.get("_last_voice", ""):
    st.session_state["_last_voice"] = voice_text
    st.session_state["_pending_voice"] = voice_text

# 3. Hidden file uploader — + button JS clicks this
inline_files = st.file_uploader(
    "chat_attach",
    type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key=f"chat_attach_{st.session_state.inline_upload_key}",
    label_visibility="hidden",
)
if inline_files:
    if not st.session_state.rag:
        st.warning("Add Groq API key first!")
    else:
        for uf in inline_files:
            already = any(d["name"] == uf.name for d in st.session_state.uploaded_docs)
            if not already:
                ext = uf.name.rsplit(".", 1)[-1].lower()
                if ext in ("pdf", "docx", "doc", "txt"):
                    with st.spinner(f"Indexing {uf.name}…"):
                        try:
                            fb = uf.read()
                            res = st.session_state.rag.add_document(fb, uf.name)
                            st.session_state.uploaded_docs.append(
                                {"name": uf.name, "chunks": res["chunks"],
                                 "size_kb": round(len(fb) / 1024, 1)})
                            st.session_state.sources_loaded.append({"type": "doc", "src": uf.name})
                            st.success(f"✓ **{uf.name}** indexed ({res['chunks']} chunks)")
                        except Exception as e:
                            st.error(f"❌ {uf.name}: {e}")
                else:
                    # Image — store in session for next query context
                    import base64
                    fb = uf.read()
                    st.session_state.setdefault("pending_images", []).append({
                        "name": uf.name,
                        "b64": base64.b64encode(fb).decode(),
                        "mime": f"image/{ext if ext != 'jpg' else 'jpeg'}"
                    })
                    st.success(f"🖼️ **{uf.name}** attached — ask a question about it!")
    st.session_state.inline_upload_key += 1
    st.rerun()

# 4. Floating + (attach) and mic buttons sitting inside the chat_input bar
components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; overflow: hidden; }

.fab {
  position: fixed;
  bottom: 16px;
  z-index: 9999;
  width: 34px; height: 34px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.15s, box-shadow 0.15s;
}
#plus-fab {
  left: 272px;   /* just right of sidebar edge, left side of chat bar */
  background: #1a1a2e;
  box-shadow: 0 0 0 1.5px #7c3aed66;
  color: #a78bfa;
  font-size: 22px; font-weight: 300; line-height: 1;
}
#plus-fab:hover { background: #2d1f5e; box-shadow: 0 0 0 2px #a78bfa; }

#mic-fab {
  right: 62px;
  background: #1a1a2e;
  box-shadow: 0 0 0 1.5px #7c3aed66;
}
#mic-fab:hover { background: #2d1f5e; box-shadow: 0 0 0 2px #a78bfa; }
#mic-fab svg { width: 16px; height: 16px; stroke: #a78bfa; }

.listening {
  background: #2d0a0a !important;
  box-shadow: 0 0 0 2px #f87171 !important;
  animation: pulse 1s infinite;
}
.listening svg { stroke: #f87171 !important; }
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 2px rgba(248,113,113,.8); }
  50%      { box-shadow: 0 0 0 7px rgba(248,113,113,0); }
}

#toast {
  position: fixed; bottom: 58px; right: 14px;
  background: #161b27; border: 1px solid #7c3aed55; color: #c084fc;
  font-size: 11px; font-family: monospace;
  padding: 5px 12px; border-radius: 20px;
  opacity: 0; transition: opacity 0.3s; pointer-events: none;
  max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  z-index: 9999;
}
.show { opacity: 1 !important; }
</style>
</head>
<body>

<!-- + attach button -->
<button id="plus-fab" class="fab" title="Attach PDF / DOCX / image" onclick="openPicker()">+</button>

<!-- mic button -->
<button id="mic-fab" class="fab" title="Voice input" onclick="toggleMic()">
  <svg viewBox="0 0 24 24" fill="none" stroke-width="2"
       stroke-linecap="round" stroke-linejoin="round">
    <rect x="9" y="2" width="6" height="12" rx="3"/>
    <path d="M19 10a7 7 0 01-14 0"/>
    <line x1="12" y1="19" x2="12" y2="22"/>
    <line x1="8"  y1="22" x2="16" y2="22"/>
  </svg>
</button>

<div id="toast"></div>

<script>
const toast = document.getElementById('toast');
const mic   = document.getElementById('mic-fab');

function showToast(msg, dur=2500) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toast.classList.remove('show'), dur);
}

/* ── + Attach: click the hidden Streamlit file input ── */
function openPicker() {
  const inputs = window.parent.document.querySelectorAll('input[type="file"]');
  if (inputs.length) {
    inputs[inputs.length - 1].click();
  } else {
    showToast('⚠ File picker not ready yet');
  }
}

/* ── Mic / Speech recognition ── */
let recog = null, listening = false;

function setVoiceInput(text) {
  const inputs = window.parent.document.querySelectorAll('input');
  for (const inp of inputs) {
    if (inp.getAttribute('aria-label') === 'voice_hidden') {
      const setter = Object.getOwnPropertyDescriptor(
        window.parent.HTMLInputElement.prototype, 'value').set;
      setter.call(inp, text);
      inp.dispatchEvent(new Event('input',  { bubbles: true }));
      inp.dispatchEvent(new Event('change', { bubbles: true }));
      break;
    }
  }
}

function toggleMic() {
  const SR = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition;
  if (!SR) { showToast('❌ Needs Chrome / Edge'); return; }
  if (listening) { recog.stop(); return; }

  recog = new SR();
  recog.continuous = false; recog.interimResults = true; recog.lang = 'hi-IN';

  recog.onstart  = () => { listening = true;  mic.classList.add('listening'); showToast('🔴 Listening…', 9999); };
  recog.onend    = () => { listening = false; mic.classList.remove('listening'); };
  recog.onerror  = (e) => { listening = false; mic.classList.remove('listening'); if (e.error !== 'no-speech') showToast('❌ ' + e.error); };
  recog.onresult = (e) => {
    let out = '';
    for (const r of e.results) out += r[0].transcript;
    if (e.results[e.results.length - 1].isFinal) {
      setVoiceInput(out);
      showToast('✓ ' + out.slice(0, 50) + (out.length > 50 ? '…' : ''));
    }
  };
  recog.start();
}
</script>
</body>
</html>
""", height=0, scrolling=False)

# 5. Main chat input — single visible bar
prompt = st.chat_input("Message…")

# Pick up voice transcript
if not prompt and st.session_state.get("_pending_voice"):
    prompt = st.session_state.pop("_pending_voice")


# ── Process prompt ────────────────────────────────────────────────────────────
def run_query(user_prompt: str):
    """Core RAG query handler — natural, context-aware conversation."""
    if not st.session_state.rag:
        st.error("Please add your Groq API key in the sidebar first!")
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result   = st.session_state.rag.query(user_prompt)
                answer   = result["answer"]
                sources  = result.get("sources", [])
                used_web = result.get("used_web_search", False)
                lang     = result.get("language_detected", "english")

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "language_detected": lang,
                })
                st.rerun()

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role":"assistant","content":err})


if prompt:
    run_query(prompt)
