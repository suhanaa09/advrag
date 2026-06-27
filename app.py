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
.main .block-container { padding-bottom: 110px !important; }
/* Hide default chat input */
[data-testid="stChatInput"] { display: none !important; }
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
        "pending_prompt": "",
        "inline_upload_key": 0,
        "bar_result": None,
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

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            badges = []
            if msg.get("used_web_search"):
                badges.append('<span class="badge-web">🌐 Live web search</span>')
            if msg.get("language_detected") and msg["language_detected"] != "english":
                badges.append(f'<span class="badge-lang">{LANG_EMOJI.get(msg["language_detected"],"")}</span>')
            if badges:
                st.markdown(" ".join(badges), unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)


# ── Custom input bar via st.components.v1.html ───────────────────────────────
# This renders in its own iframe — scripts work correctly here
bar_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Space+Mono&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d0f14;
    font-family: 'Inter', sans-serif;
    padding: 10px 12px 12px 12px;
  }

  /* File pill strip */
  #file-strip {
    display: none; flex-wrap: wrap; gap: 6px; margin-bottom: 8px;
  }
  .file-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: #1a1040; border: 1px solid #7c3aed66;
    color: #c084fc; border-radius: 20px; padding: 3px 10px;
    font-size: 12px; font-family: 'Space Mono', monospace;
  }
  .file-pill .rm {
    cursor: pointer; color: #94a3b8; margin-left: 3px;
    font-size: 15px; line-height: 1;
  }
  .file-pill .rm:hover { color: #f87171; }

  /* Bar */
  #bar {
    display: flex; align-items: flex-end; gap: 6px;
    background: #161b27;
    border: 1.5px solid #7c3aed;
    border-radius: 14px;
    padding: 8px 10px;
    transition: border-color 0.2s;
  }
  #bar:focus-within { border-color: #a78bfa; }

  /* Buttons */
  .icon-btn {
    background: none; border: none; cursor: pointer;
    color: #64748b; padding: 5px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    transition: color 0.15s, background 0.15s; flex-shrink: 0;
  }
  .icon-btn:hover { color: #a78bfa; background: #7c3aed22; }
  .icon-btn svg { width: 20px; height: 20px; }

  /* Mic active */
  .mic-on { color: #f87171 !important; background: #7c3aed22 !important; }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(248,113,113,.5); }
    50%      { box-shadow: 0 0 0 7px rgba(248,113,113,0); }
  }
  .mic-on { animation: pulse 1s ease infinite; }

  /* Textarea */
  #ta {
    flex: 1; background: transparent; border: none; outline: none;
    color: #e2e8f0; font-family: 'Inter', sans-serif; font-size: 15px;
    resize: none; min-height: 24px; max-height: 130px;
    line-height: 1.5; padding: 2px 4px; overflow-y: auto;
  }
  #ta::placeholder { color: #475569; }

  /* Send */
  #send-btn {
    background: linear-gradient(135deg,#7c3aed,#4f46e5);
    border: none; border-radius: 10px; cursor: pointer;
    color: white; padding: 7px 13px; display: flex;
    align-items: center; justify-content: center;
    transition: opacity .15s; flex-shrink: 0;
  }
  #send-btn:hover { opacity: .85; }
  #send-btn svg { width: 17px; height: 17px; }

  input[type=file] { display: none; }
</style>
</head>
<body>

<!-- File preview -->
<div id="file-strip"></div>

<!-- Bar -->
<div id="bar">

  <!-- Attach -->
  <button class="icon-btn" title="Attach PDF / DOCX / TXT"
          onclick="document.getElementById('fi').click()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19
               a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
    </svg>
  </button>
  <input type="file" id="fi" multiple accept=".pdf,.docx,.doc,.txt"
         onchange="onFiles(event)">

  <!-- Textarea -->
  <textarea id="ta" rows="1"
    placeholder="Ask in English, Hindi, या Hinglish…"
    oninput="resize(this)" onkeydown="onKey(event)"></textarea>

  <!-- Mic -->
  <button class="icon-btn" id="mic" title="Voice input" onclick="toggleMic()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="2" width="6" height="12" rx="3"/>
      <path d="M19 10a7 7 0 01-14 0"/>
      <line x1="12" y1="19" x2="12" y2="22"/>
      <line x1="8"  y1="22" x2="16" y2="22"/>
    </svg>
  </button>

  <!-- Send -->
  <button id="send-btn" title="Send" onclick="send()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  </button>
</div>

<script>
// ── Resize textarea ──────────────────────────────────────────────────────────
function resize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

// Enter = send, Shift+Enter = newline
function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

// ── Files ────────────────────────────────────────────────────────────────────
let files = [];

function onFiles(e) {
  Array.from(e.target.files).forEach(f => {
    if (!files.find(p => p.name === f.name)) files.push(f);
  });
  e.target.value = '';
  renderPills();
}

function renderPills() {
  const strip = document.getElementById('file-strip');
  strip.innerHTML = '';
  if (!files.length) { strip.style.display = 'none'; return; }
  strip.style.display = 'flex';
  files.forEach((f, i) => {
    const icons = {pdf:'📕',docx:'📘',doc:'📘',txt:'📄'};
    const ext   = f.name.split('.').pop().toLowerCase();
    const label = f.name.length > 26 ? f.name.slice(0,23)+'…' : f.name;
    const pill  = document.createElement('span');
    pill.className = 'file-pill';
    pill.innerHTML = `${icons[ext]||'📎'} ${label}
      <span class="rm" onclick="rmFile(${i})">×</span>`;
    strip.appendChild(pill);
  });
}

function rmFile(i) { files.splice(i,1); renderPills(); }

// ── Send ─────────────────────────────────────────────────────────────────────
async function send() {
  const ta   = document.getElementById('ta');
  const text = ta.value.trim();

  // Convert files to base64
  let filesPayload = [];
  for (const f of files) {
    const b64 = await new Promise((res, rej) => {
      const r = new FileReader();
      r.onload  = () => res(r.result.split(',')[1]);
      r.onerror = rej;
      r.readAsDataURL(f);
    });
    filesPayload.push({ name: f.name, data: b64 });
  }

  if (!text && !filesPayload.length) return;

  // Send to Streamlit parent via postMessage
  window.parent.postMessage({
    type: 'streamlit:setComponentValue',
    value: { text: text, files: filesPayload }
  }, '*');

  ta.value = '';
  ta.style.height = 'auto';
  files = [];
  renderPills();
}

// ── Mic ──────────────────────────────────────────────────────────────────────
let recog = null, listening = false;

function toggleMic() {
  const btn    = document.getElementById('mic');
  const SR     = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert('Speech recognition needs Chrome or Edge.'); return; }

  if (listening) { recog.stop(); return; }

  recog = new SR();
  recog.continuous     = false;
  recog.interimResults = true;
  recog.lang           = 'hi-IN';   // handles Hindi + Hinglish + English

  recog.onstart  = () => { listening = true;  btn.classList.add('mic-on');    btn.title='Listening… click to stop'; };
  recog.onend    = () => { listening = false; btn.classList.remove('mic-on'); btn.title='Voice input'; };
  recog.onerror  = (e) => {
    listening = false; btn.classList.remove('mic-on');
    if (e.error !== 'no-speech') alert('Mic error: ' + e.error);
  };
  recog.onresult = (e) => {
    let out = '';
    for (const r of e.results) out += r[0].transcript;
    document.getElementById('ta').value = out;
    resize(document.getElementById('ta'));
  };

  recog.start();
}
</script>
</body>
</html>
"""

# Render the bar as an iframe component — scripts work here
bar_result = components.html(bar_html, height=90, scrolling=False)

# ── Process inline quick-upload files from bar ────────────────────────────────
# Because components.html() can't directly trigger a Streamlit rerun with file data,
# we use a hidden file_uploader as the actual upload bridge for the bar's 📎 button.
# The bar's mic/text goes through st.chat_input which we keep hidden then re-enable.

# Inline file uploader — triggered when user picks files in the custom bar.
# We keep it visible but styled minimally below the bar as a fallback upload zone.
st.markdown("""
<style>
/* Style the fallback uploader as a slim strip */
div[data-testid="stFileUploaderDropzoneInstructions"] > div > span { font-size:12px !important; }
section[data-testid="stFileUploaderDropzone"] {
    padding: 6px 12px !important; min-height: unset !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

quick_files = st.file_uploader(
    "📎 Drop files here or use the 📎 button above",
    type=["pdf","docx","doc","txt"],
    accept_multiple_files=True,
    key=f"qf_{st.session_state.inline_upload_key}",
)

if quick_files:
    if not st.session_state.rag:
        st.error("Add Groq API key first!")
    else:
        for uf in quick_files:
            already = any(d["name"]==uf.name for d in st.session_state.uploaded_docs)
            if not already:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        fb = uf.read()
                        res = st.session_state.rag.add_document(fb, uf.name)
                        st.session_state.uploaded_docs.append(
                            {"name":uf.name,"chunks":res["chunks"],"size_kb":round(len(fb)/1024,1)})
                        st.session_state.sources_loaded.append({"type":"doc","src":uf.name})
                        st.success(f"✓ **{uf.name}** — {res['chunks']} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")
        st.session_state.inline_upload_key += 1
        st.rerun()

# ── Text input via st.chat_input ─────────────────────────────────────────────
# We use the real st.chat_input but hide it — the custom bar above sends
# the text through query_params when the user hits send.
# For the text path, use st.chat_input which is fully functional.
prompt = st.chat_input("Ask in English, Hindi, या Hinglish…")

if prompt:
    if not st.session_state.rag:
        st.error("Please add your Groq API key in the sidebar first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result   = st.session_state.rag.query(prompt)
                answer   = result["answer"]
                sources  = result.get("sources", [])
                used_web = result.get("used_web_search", False)
                lang     = result.get("language_detected", "english")

                st.markdown(answer)

                badges = []
                if used_web:
                    badges.append('<span class="badge-web">🌐 Live web search</span>')
                if lang != "english":
                    badges.append(f'<span class="badge-lang">{LANG_EMOJI.get(lang,"")}</span>')
                if badges:
                    st.markdown(" ".join(badges), unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for src in sources:
                            st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "used_web_search": used_web,
                    "language_detected": lang,
                })
                st.rerun()

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role":"assistant","content":err})
