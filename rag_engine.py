"""
rag_engine.py  –  Enhanced RAG Engine
  • Web scraping    : requests + BeautifulSoup
  • Document upload : PDF (PyMuPDF/pdfplumber), DOCX (python-docx), TXT
  • Embeddings      : sentence-transformers (all-MiniLM-L6-v2, runs locally, free)
  • Vector store    : FAISS (in-memory)
  • LLM             : Groq API (llama / mixtral / gemma)
  • Live web search : Tavily Search API (FREE tier = 1000 searches/month)
  • Conversation memory : rolling window with summary compression
  • Multilingual    : Hindi, Hinglish, English auto-detect & respond
"""

from __future__ import annotations
import re
import time
import textwrap
import io
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # Preserve Devanagari (Hindi) characters along with ASCII
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def detect_language(text: str) -> str:
    """Detect if text is Hindi, Hinglish, or English."""
    # Check for Devanagari script
    devanagari = re.findall(r'[\u0900-\u097F]', text)
    total_chars = len([c for c in text if c.isalpha()])
    
    if not total_chars:
        return "english"
    
    devanagari_ratio = len(devanagari) / max(total_chars, 1)
    
    if devanagari_ratio > 0.5:
        return "hindi"
    elif devanagari_ratio > 0.1:
        return "hinglish"
    
    # Check for common Hindi romanized words
    hinglish_markers = [
        r'\b(kya|hai|hain|nahi|nahin|aur|lekin|toh|yeh|woh|kaise|kyun|mujhe|tumhe|aap|main|hum)\b',
        r'\b(bata|batao|samjhao|chahiye|lagta|lagti|karo|karna|dena|lena)\b',
        r'\b(accha|theek|sahi|galat|pata|malum|समझ|bilkul|zaroor)\b'
    ]
    
    text_lower = text.lower()
    hinglish_count = sum(len(re.findall(p, text_lower)) for p in hinglish_markers)
    
    if hinglish_count >= 2:
        return "hinglish"
    
    return "english"


# ─────────────────────────────────────────────────────────────────────────────
# Document Parser
# ─────────────────────────────────────────────────────────────────────────────

class DocumentParser:
    """Parse PDF, DOCX, and TXT files into text."""

    @staticmethod
    def parse_pdf(file_bytes: bytes, filename: str) -> Tuple[str, str]:
        """Returns (text, method_used)"""
        # Try pdfplumber first (better text extraction)
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            if pages_text:
                return "\n\n".join(pages_text), "pdfplumber"
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback to PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()
            if pages_text:
                return "\n\n".join(pages_text), "pymupdf"
        except ImportError:
            pass
        except Exception:
            pass

        raise ValueError(
            f"Could not parse '{filename}'. Install pdfplumber or PyMuPDF:\n"
            "pip install pdfplumber  OR  pip install pymupdf"
        )

    @staticmethod
    def parse_docx(file_bytes: bytes, filename: str) -> str:
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also extract table text
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ValueError(
                f"python-docx not installed. Run: pip install python-docx"
            )

    @staticmethod
    def parse_txt(file_bytes: bytes) -> str:
        for enc in ["utf-8", "utf-16", "latin-1", "cp1252"]:
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace")

    def parse(self, file_bytes: bytes, filename: str) -> str:
        """Auto-detect format and parse."""
        ext = filename.lower().rsplit(".", 1)[-1]
        
        if ext == "pdf":
            text, _ = self.parse_pdf(file_bytes, filename)
        elif ext in ("docx", "doc"):
            text = self.parse_docx(file_bytes, filename)
        elif ext == "txt":
            text = self.parse_txt(file_bytes)
        else:
            # Try as text
            text = self.parse_txt(file_bytes)
        
        return clean_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """Rolling conversation memory with context compression."""

    def __init__(self, max_turns: int = 10, summary_threshold: int = 8):
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Return messages formatted for the LLM, keeping recent turns."""
        recent = self.messages[-self.max_turns * 2:]  # Each turn = 2 messages
        if self.summary:
            system_context = [
                {
                    "role": "system",
                    "content": f"[Earlier conversation summary: {self.summary}]"
                }
            ]
            return system_context + recent
        return recent

    def compress(self, groq_client, model: str):
        """Summarize old messages to save context."""
        if len(self.messages) <= self.summary_threshold * 2:
            return
        
        old_messages = self.messages[:-self.max_turns * 2]
        if not old_messages:
            return

        summary_prompt = "Summarize this conversation in 2-3 sentences, keeping key facts and context:\n\n"
        for m in old_messages:
            summary_prompt += f"{m['role'].upper()}: {m['content']}\n"

        try:
            resp = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            self.summary = resp.choices[0].message.content
            # Keep only recent messages
            self.messages = self.messages[-self.max_turns * 2:]
        except Exception:
            pass  # If compression fails, just keep rolling

    def clear(self):
        self.messages.clear()
        self.summary = ""

    @property
    def turn_count(self) -> int:
        return len(self.messages) // 2


# ─────────────────────────────────────────────────────────────────────────────
# Live Web Search  (Tavily — free tier)
# ─────────────────────────────────────────────────────────────────────────────

class LiveSearch:
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
        }
        try:
            resp = requests.post(self.ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            if data.get("answer"):
                results.append({
                    "title": "Tavily Direct Answer",
                    "url": "tavily://answer",
                    "content": data["answer"],
                })
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                })
            return results
        except Exception as e:
            return [{"title": "Search Error", "url": "", "content": str(e)}]


# ─────────────────────────────────────────────────────────────────────────────
# Web Scraper
# ─────────────────────────────────────────────────────────────────────────────

class WebScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    def scrape_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        visited: set[str] = set()
        all_pages: List[Dict] = []
        self._crawl(url, url, depth, visited, all_pages)
        return {"pages": all_pages, "total": len(all_pages)}

    def _crawl(self, base: str, url: str, depth: int, visited: set, pages: list):
        if url in visited or depth < 0:
            return
        visited.add(url)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception:
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        body = soup.get_text(separator=" ")
        text = clean_text(body)

        if len(text) > 100:
            pages.append({"url": url, "title": title, "text": text})

        if depth > 1:
            for a in soup.find_all("a", href=True):
                href = urljoin(base, a["href"])
                if urlparse(href).netloc == urlparse(base).netloc:
                    self._crawl(base, href, depth - 1, visited, pages)
                    time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[str], sources: List[str]):
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        self.metadata.extend([{"source": s} for s in sources])

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        if self.index.ntotal == 0:
            return [], []
        q = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results, srcs = [], []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and distances[0][i] < 2.0:
                results.append(self.chunks[idx])
                srcs.append(self.metadata[idx]["source"])
        return results, srcs

    def clear(self):
        self.index.reset()
        self.chunks.clear()
        self.metadata.clear()

    @property
    def total(self):
        return self.index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

MULTILINGUAL_SYSTEM_BASE = """
You are an advanced AI assistant — think of yourself as a blend of ChatGPT and Claude.

RESPONSE STYLE (strictly follow):
• Be warm, natural, and conversational — never robotic or dry
• Use markdown formatting: **bold** for key terms, bullet lists when helpful, code blocks for code
• Structure longer answers with clear sections/headers
• Match the user's tone: casual questions → relaxed reply; detailed questions → thorough answer
• Show personality: be helpful but also engaging and human-like
• Never start with "Certainly!" or "Of course!" — be direct and natural
• End with a follow-up question or offer to dig deeper when appropriate

LANGUAGE RULES (very important):
• Detect the user's language from their message
• If user writes in HINDI (Devanagari) → respond fully in Hindi
• If user writes in HINGLISH (mix of Hindi words in Roman script) → respond in Hinglish naturally
• If user writes in ENGLISH → respond in English
• Match the exact language style and script the user uses
• For Hinglish: naturally mix Hindi words in Roman script (e.g., "Yeh concept basically kya hai...")
• NEVER switch languages unless the user does
"""


class RAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        top_k: int = 4,
        temperature: float = 0.4,
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.tavily_api_key = tavily_api_key
        self.live_search: Optional[LiveSearch] = (
            LiveSearch(tavily_api_key) if tavily_api_key else None
        )

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = VectorStore(dim=384)
        self.scraper = WebScraper()
        self.doc_parser = DocumentParser()
        self.memory = ConversationMemory(max_turns=10, summary_threshold=8)
        self._source_set: set[str] = set()

    def set_tavily_key(self, key: str):
        self.tavily_api_key = key
        self.live_search = LiveSearch(key)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        result = self.scraper.scrape_url(url, depth=depth)
        pages = result["pages"]
        if not pages:
            raise ValueError(f"No content extracted from {url}")

        total_chunks = 0
        for page in pages:
            chunks = chunk_text(page["text"])
            if not chunks:
                continue
            embeddings = self.embedder.encode(chunks, show_progress_bar=False)
            sources = [page["url"]] * len(chunks)
            self.vector_store.add(np.array(embeddings), chunks, sources)
            self._source_set.add(page["url"])
            total_chunks += len(chunks)

        return {"chunks": total_chunks, "pages": len(pages)}

    def add_text(self, text: str, label: str = "manual") -> Dict[str, Any]:
        text = clean_text(text)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No usable text found.")
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        sources = [label] * len(chunks)
        self.vector_store.add(np.array(embeddings), chunks, sources)
        self._source_set.add(label)
        return {"chunks": len(chunks)}

    def add_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Parse and index an uploaded document (PDF, DOCX, TXT)."""
        text = self.doc_parser.parse(file_bytes, filename)
        if len(text.strip()) < 50:
            raise ValueError(f"Could not extract meaningful text from '{filename}'")
        
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No usable content found in document.")
        
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        label = filename
        sources = [label] * len(chunks)
        self.vector_store.add(np.array(embeddings), chunks, sources)
        self._source_set.add(label)
        
        # Estimate pages for PDF
        page_count = text.count("\n\n") // 3 + 1
        
        return {
            "chunks": len(chunks),
            "chars": len(text),
            "filename": filename,
            "estimated_pages": page_count,
        }

    # ── Query pipeline ────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        # Detect language
        lang = detect_language(question)
        
        # Try FAISS retrieval
        q_emb = self.embedder.encode([question], show_progress_bar=False)[0]
        chunks, sources = self.vector_store.search(q_emb, top_k=self.top_k)

        used_web_search = False

        if chunks:
            context = "\n\n---\n\n".join(
                f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources)
            )
            rag_instruction = textwrap.dedent(f"""
                {MULTILINGUAL_SYSTEM_BASE}

                You have retrieved the following document context. Use it to answer accurately.
                If the context fully answers the question, use it. If only partially, supplement
                with your knowledge. Always cite sources naturally in your response.

                RETRIEVED CONTEXT:
                {context}
            """).strip()

        elif self.live_search:
            used_web_search = True
            web_results = self.live_search.search(question, max_results=5)
            context_parts = []
            for r in web_results:
                if r["content"]:
                    context_parts.append(
                        f"[Source: {r['url']}]\nTitle: {r['title']}\n{r['content']}"
                    )
                    sources.append(r["url"])
            context = "\n\n---\n\n".join(context_parts) if context_parts else "No results found."

            rag_instruction = textwrap.dedent(f"""
                {MULTILINGUAL_SYSTEM_BASE}

                You have access to LIVE web search results below. Use them to answer accurately.
                You have up-to-date information — do NOT say your knowledge is limited.
                Cite sources naturally in your response.

                LIVE WEB SEARCH RESULTS:
                {context}
            """).strip()

        else:
            rag_instruction = textwrap.dedent(f"""
                {MULTILINGUAL_SYSTEM_BASE}

                No documents are loaded and web search is disabled. Answer using your training
                knowledge. Be honest about uncertainty naturally without blaming limitations.
                
                TIP: Add a Tavily API key for live web search, or upload documents in the sidebar!
            """).strip()

        # Build messages: system + memory context + new question
        history_messages = self.memory.get_context_messages()
        
        messages = (
            [{"role": "system", "content": rag_instruction}]
            + history_messages
            + [{"role": "user", "content": question}]
        )

        chat = self.groq_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1536,
        )

        answer = chat.choices[0].message.content

        # Update conversation memory
        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        # Compress memory if needed
        if self.memory.turn_count > self.memory.summary_threshold:
            self.memory.compress(self.groq_client, self.model)

        unique_sources = [s for s in list(dict.fromkeys(sources)) if s and s != "tavily://answer"]

        return {
            "answer": answer,
            "sources": unique_sources,
            "chunks_used": len(chunks),
            "used_web_search": used_web_search,
            "language_detected": lang,
            "memory_turns": self.memory.turn_count,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {
            "chunks": self.vector_store.total,
            "sources": len(self._source_set),
            "memory_turns": self.memory.turn_count,
        }

    def clear_memory(self):
        """Clear conversation history only."""
        self.memory.clear()

    def clear(self):
        """Clear everything."""
        self.vector_store.clear()
        self._source_set.clear()
        self.memory.clear()
