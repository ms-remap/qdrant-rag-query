import time
from typing import List, Tuple, Optional, Literal

import traceback
import requests
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os

# ============================================================
# ======================== CONFIG =============================
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "intfloat/multilingual-e5-large"
)

LLM_URL = os.getenv(
    "LLM_URL",
    "http://localhost:2146/v1/chat/completions"
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-120b"
)

# Retrieval settings
DENSE_LIMIT = int(os.getenv("DENSE_LIMIT", 200))
SPARSE_LIMIT = int(os.getenv("SPARSE_LIMIT", 400))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", 150))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 500_000))



# ============================================================
# ======================== LOGGING ============================
# ============================================================

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_step(msg: str):
    print(f"[{_now_str()}][QUERY][INFO] {msg}", flush=True)


def log_error(msg: str):
    print(f"[{_now_str()}][QUERY][ERROR] {msg}", flush=True)


def log_exception(prefix: str, exc: Exception):
    log_error(f"{prefix}: {exc}")
    tb = traceback.format_exc()
    print(f"[{_now_str()}][QUERY][TRACEBACK]\n{tb}", flush=True)


# ============================================================
# ==================== CLIENTS INIT ===========================
# ============================================================

log_step("Loading dense embedding model...")
try:
    _dense_embedder = SentenceTransformer(EMBED_MODEL)
    log_step("Dense model ready.")
except Exception as e:
    log_exception("Failed to load dense embedding model", e)
    raise

log_step("Loading BM25 sparse model (FastEmbed)...")
try:
    _SparseTextEmbeddingClass = SparseTextEmbedding
    _sparse_bm25 = _SparseTextEmbeddingClass("Qdrant/bm25")
    log_step("BM25 sparse model ready.")
except Exception as e:
    log_exception("Failed to load BM25 sparse model", e)
    raise

log_step("Initializing Qdrant client...")
try:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,  # suppress version warning if versions differ
    )
    log_step("Qdrant client ready.")
except Exception as e:
    log_exception("Failed to initialize Qdrant client", e)
    raise


# ============================================================
# ==================== EMAIL → COLLECTION =====================
# ============================================================

def email_to_collection(user_email: str) -> str:
    """
    Map a mailbox email (e.g. 'owasi@remap.ai') to the per-user collection name
    used by the ETL, e.g. 'email_owasi_remap_ai'.
    """
    log_step(f"email_to_collection() called with user_email={user_email!r}")
    if not user_email:
        msg = "user_email is required to route query to correct collection"
        log_error(msg)
        raise ValueError(msg)

    base = user_email.strip().lower()
    safe = "".join(c if c.isalnum() else "_" for c in base)
    collection_name = f"email_{safe}"
    log_step(f"Mapped user_email={user_email} -> collection={collection_name}")
    return collection_name


def ensure_collection_exists(collection_name: str):
    """
    Ensure that the per-user collection exists in Qdrant.
    Raise a clear error if it does not.
    """
    log_step(f"ensure_collection_exists() checking collection={collection_name}")
    try:
        coll_info = qdrant.get_collection(collection_name)
        log_step(
            f"Using existing Qdrant collection: {collection_name} | "
            f"status={getattr(coll_info, 'status', 'unknown')}"
        )
    except Exception as e:
        log_exception(f"Qdrant collection '{collection_name}' not found or inaccessible", e)
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' not found. "
            f"Run the ETL for this mailbox first."
        ) from e


# ============================================================
# ===================== SMALL TALK CHECK =====================
# ============================================================

def is_small_talk(query: str) -> bool:
    """
    Simple small-talk / greeting detection.
    If this returns True, we skip Qdrant completely and just chat.
    """
    log_step(f"is_small_talk() called with query={query!r}")
    q = (query or "").lower().strip()
    if not q:
        log_step("is_small_talk() -> False (empty query after strip)")
        return False

    phrases = [
        "hi", "hello", "hey", "yo",
        "salam", "salaam", "assalam o alaikum", "assalamu alaikum",
        "how are you", "how r u", "kese ho", "kaisay ho", "kia hal hai", "kya haal hai",
    ]

    for p in phrases:
        if q == p or q.startswith(p):
            log_step(f"is_small_talk() -> True (matched phrase={p!r})")
            return True

    # Very short messages that look like greeting
    if len(q.split()) <= 3 and any(w in q for w in ["hi", "hello", "hey"]):
        log_step("is_small_talk() -> True (short greeting-like message)")
        return True

    log_step("is_small_talk() -> False")
    return False


# ============================================================
# ====================== LLM HELPER ==========================
# ============================================================

def _llm_chat(messages, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Low-level helper to call local OpenAI-compatible LLM.
    Logs the full payload going out and some details coming back.

    NOTE: gpt-oss uses a reasoning channel internally. We control that via
    the prompts (e.g. "Reasoning: low") and by keeping max_tokens large
    enough so it can finish and write to message.content.
    """
    log_step(
        f"_llm_chat() called with temperature={temperature}, max_tokens={max_tokens}, "
        f"num_messages={len(messages)}"
    )
    # if not (LLM_URL and LLM_API_KEY and LLM_MODEL):
    #     msg = "LLM config missing (LLM_URL / LLM_API_KEY / LLM_MODEL)."
    #     log_error(msg)
    #     raise RuntimeError(msg)

    # Optional: log first and last message for visibility
    if messages:
        log_step(
            f"_llm_chat() first message role={messages[0]['role']}, "
            f"content_preview={messages[0]['content'][:200]!r}"
        )
        log_step(
            f"_llm_chat() last message role={messages[-1]['role']}, "
            f"content_preview={messages[-1]['content'][:200]!r}"
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    log_step(f"Calling LLM at {LLM_URL} with model={LLM_MODEL}")
    try:
        r = requests.post(LLM_URL, headers=headers, json=body, timeout=120)
        log_step(f"LLM HTTP status={r.status_code}")
        r.raise_for_status()
        data = r.json()
        log_step(f"LLM response JSON keys={list(data.keys())}")
    except Exception as e:
        log_exception("LLM HTTP error", e)
        raise

    try:
        choice = data["choices"][0]["message"]
        content = choice.get("content")
        if content is None:
            # Avoid the crash you saw (len(None)).
            # We log it clearly so you can debug the model config.
            log_error(
                "_llm_chat(): message.content is None. "
                "This usually means the model is stuck in reasoning-only mode "
                "or max_tokens is too low. Returning empty string."
            )
            content = ""
        log_step(f"_llm_chat() received content length={len(content)}")
    except Exception as e:
        log_error(f"Unexpected LLM response structure: {data}")
        raise RuntimeError(f"Unexpected LLM response: {e}") from e

    return content


# ============================================================
# =========== QUERY TYPE CLASSIFIER (SMART / CONTEXTUAL) =====
# ============================================================

CLASSIFIER_SYSTEM_PROMPT = """
Reasoning: low
You are an intent classifier and router for an Outlook email assistant.

You NEVER answer the user directly.
You ONLY inspect:
- The recent chat history.
- The user's CURRENT query.

You must output EXACTLY one JSON object, with this schema:

{
  "query_type": "...",
  "needs_email_rag": true/false
}

Where:

- "query_type" MUST be one of:
  - "small_talk"           → greetings, thanks, casual chat, meta questions NOT about emails.
  - "email_search"         → user wants you to FIND/LIST emails from the mailbox
                             (by sender, subject, date, keyword, etc.).
  - "email_summarization"  → user wants summaries, key points, explanations, or comparisons
                             of specific emails (already visible in chat OR to be fetched).
  - "email_composition"    → user wants to write / draft / reply to an email.
  - "config_or_debug"      → user is talking about bugs, errors, logs, or configuration
                             of the assistant / pipeline (N8N, Qdrant, LLM, etc.).
  - "other"                → anything that doesn’t fit the above.

- "needs_email_rag":
  - true  → you must search/read email data (mailbox / Qdrant / Outlook API).
  - false → you can answer purely from existing chat history + general knowledge.

====================
INTENT RULES
====================

Use BOTH the history and the CURRENT query when classifying.

1) SMALL TALK
   - If the query is just greetings, thanks, or chitchat:
     {
       "query_type": "small_talk",
       "needs_email_rag": false
     }

2) EMAIL SEARCH (fetch emails)
   - User asks to "find", "search", "show", "list" or "look up" emails,
     often with filters (sender, date, subject, keyword).
   - Examples:
       "find emails from John last week"
       "show invoices from January"
   - Output:
     {
       "query_type": "email_search",
       "needs_email_rag": true
     }

3) EMAIL SUMMARIZATION (summaries, key points)
   - The user wants summaries/key points of EMAILS.

   a) If the relevant emails (subjects, tables, previous summaries) are already
      displayed in the recent assistant messages, and the user refers to them
      with pronouns or deictic phrases like:
        - "them", "those", "these", "that one"
        - "the 3 emails you just showed me", etc.
      Then you can work from history alone:
        {
          "query_type": "email_summarization",
          "needs_email_rag": false
        }

   b) If the user clearly wants summaries of emails that are NOT yet shown
      in the chat history, e.g.:
        - "summarize all my unread emails from yesterday"
        - "give me key points from my last 10 emails"
      Then you MUST fetch emails:
        {
          "query_type": "email_summarization",
          "needs_email_rag": true
        }

4) EMAIL COMPOSITION (write or reply)
   - User wants to create or reply to an email.
   - Examples:
       "Reply to that saying thanks for the update."
       "Draft an email to Sarah about the budget."
   - If we must read a specific email to reply accurately:
        {
          "query_type": "email_composition",
          "needs_email_rag": true
        }
   - If the user provides all content themselves and no mailbox context is needed:
        {
          "query_type": "email_composition",
          "needs_email_rag": false
        }

5) CONFIG_OR_DEBUG (logs / errors / setup)
   - If the current query looks like:
       - Error messages, stack traces, logs (e.g. "(Received empty response or unknown format from N8N)")
       - Questions about "why did you show this error", "fix this integration", "change settings"
     Then:
        {
          "query_type": "config_or_debug",
          "needs_email_rag": false
        }

6) OTHER
   - For anything that doesn't fit the above categories:
        {
          "query_type": "other",
          "needs_email_rag": false
        }

====================
DISAMBIGUATION WITH HISTORY
====================

- If the CURRENT query is vague but the chat history shows that the assistant
  just listed or summarized specific emails in a table, and the user now says:
    "give me key points of them"
    "summarize those"
    "compare them"
  → Treat this as:
     {
       "query_type": "email_summarization",
       "needs_email_rag": false
     }

- If the CURRENT query is literally an error string like:
    "(Received empty response or unknown format from N8N)"
  → classify it as:
     {
       "query_type": "config_or_debug",
       "needs_email_rag": false
     }

- If the user re-sends or copies those error messages as their query,
  assume they want help with the error, NOT email search.

====================
OUTPUT FORMAT
====================

Important:
- Output MUST be ONLY the JSON object.
- No extra text, no explanations, no markdown, no backticks.

Examples of valid outputs:

{"query_type": "email_summarization", "needs_email_rag": false}
{"query_type": "email_search", "needs_email_rag": true}
{"query_type": "config_or_debug", "needs_email_rag": false}
{"query_type": "other", "needs_email_rag": false}
"""


def classify_query_with_history(
    user_query: str,
    history: Optional[List["ChatMessage"]] = None,
) -> dict:
    """
    Decide:
      - query_type: small_talk | email_search | email_summarization |
                    email_composition | config_or_debug | other
      - needs_email_rag: bool

    Uses BOTH history + current query intelligently, so that e.g.
    "give me key points of them" after we just showed 3 emails does NOT
    go to Qdrant again; it summarises from chat history.
    """
    log_step(
        f"[DEBUG] classify_query_with_history(): START user_query={user_query!r}, "
        f"history_len={len(history) if history else 0}"
    )

    # Empty query → safe default
    q = (user_query or "").strip()
    if not q:
        log_step("[DEBUG] classify_query_with_history(): empty query → other / no RAG")
        return {
            "query_type": "other",
            "needs_email_rag": False,
        }

    # Cheap rule-based short-circuit: obvious small talk
    if is_small_talk(user_query):
        log_step(
            "[DEBUG] classify_query_with_history(): is_small_talk() == True → "
            "query_type=small_talk, needs_email_rag=False"
        )
        return {
            "query_type": "small_talk",
            "needs_email_rag": False,
        }

    # History → compact text (last 10 messages)
    if history:
        trimmed = history[-10:]
        history_lines = []
        for m in trimmed:
            role = m.role.upper()
            content = m.content
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "(no previous history)"

    user_block = (
        "Recent chat history:\n"
        f"{history_text}\n\n"
        "User's current query:\n"
        f"{user_query}\n\n"
        "Return ONLY the JSON object as specified."
    )

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

    # Use a generous max_tokens so the model can finish reasoning + JSON,
    # to avoid the earlier `content=None` issue when it got cut at 128 tokens.
    try:
        raw = _llm_chat(messages, temperature=0.0, max_tokens=512).strip()
    except Exception as e:
        log_exception(
            "classify_query_with_history(): _llm_chat raised, falling back to heuristics",
            e,
        )
        raw = ""

    log_step(f"classify_query_with_history(): raw LLM output={raw!r}")

    # JSON parse + robust fallback
    try:
        parsed = json.loads(raw)
    except Exception as e:
        log_exception(
            "classify_query_with_history(): failed to parse JSON, using heuristic fallback",
            e,
        )

        lower_q = q.lower()

        # Error / logs / N8N / stack traces
        if any(tok in lower_q for tok in ["traceback", "exception", "error", "n8n"]):
            parsed = {"query_type": "config_or_debug", "needs_email_rag": False}

        # Classic email intents
        elif any(
            k in lower_q
            for k in [
                "email", "inbox", "mailbox", "subject", "from:",
                "unread", "recent emails", "last emails",
                "client", "invoice", "payment", "meeting",
            ]
        ):
            parsed = {"query_type": "email_search", "needs_email_rag": True}
        else:
            parsed = {"query_type": "other", "needs_email_rag": False}

    query_type = parsed.get("query_type", "other")
    needs_email_rag = bool(parsed.get("needs_email_rag", False))

    log_step(
        f"[DEBUG] classify_query_with_history(): final classification "
        f"query_type={query_type}, needs_email_rag={needs_email_rag}"
    )

    return {
        "query_type": query_type,
        "needs_email_rag": needs_email_rag,
    }


# ============================================================
# ==================== EMBEDDING HELPERS =====================
# ============================================================

def embed_dense(text: str) -> list:
    """Dense embedding; same model + normalize=True as ETL."""
    log_step(f"embed_dense() called, text_preview={text[:200]!r}")
    try:
        vec = _dense_embedder.encode([text], normalize_embeddings=True)[0]
    except Exception as e:
        log_exception("embed_dense() failed", e)
        raise
    dense_list = vec.tolist()
    log_step(f"embed_dense() -> vector_size={len(dense_list)}")
    return dense_list


def embed_sparse_bm25(text: str):
    """BM25 sparse embedding via FastEmbed."""
    log_step(f"embed_sparse_bm25() called, text_preview={text[:200]!r}")
    try:
        emb = list(_sparse_bm25.embed([text]))[0]
        obj = emb.as_object()  # {"indices": [...], "values": [...]}
    except Exception as e:
        log_exception("embed_sparse_bm25() failed", e)
        raise

    indices = obj["indices"]
    values = obj["values"]
    log_step(
        f"embed_sparse_bm25() -> {len(indices)} non-zero entries "
        f"(indices_sample={indices[:10]}, values_sample={values[:10]})"
    )
    return indices, values


# ============================================================
# ===================== HYBRID SEARCH ========================
# ============================================================

def hybrid_search_qdrant(
    user_query: str,
    user_email: str,
    dense_limit: int = DENSE_LIMIT,
    sparse_limit: int = SPARSE_LIMIT,
    final_top_k: int = FINAL_TOP_K,
):
    """
    Hybrid search that matches the email ETL setup:
    - dense: 'dense'
    - sparse (BM25): 'bm25'
    Uses Qdrant Query API with RRF fusion.

    Uses:
    - user_email to route to correct per-mailbox collection
    - raw user query (no spelling/grammar correction)
    """
    log_step(
        f"hybrid_search_qdrant() called with user_query={user_query!r}, "
        f"user_email={user_email!r}, dense_limit={dense_limit}, "
        f"sparse_limit={sparse_limit}, final_top_k={final_top_k}"
    )

    query = (user_query or "").strip()
    if not query:
        msg = "Empty query passed to hybrid_search_qdrant"
        log_error(msg)
        raise ValueError(msg)

    collection_name = email_to_collection(user_email)
    ensure_collection_exists(collection_name)

    dense_vec = embed_dense(query)
    log_step(f"hybrid_search_qdrant() dense vector ready, size={len(dense_vec)}")
    sparse_indices, sparse_values = embed_sparse_bm25(query)
    log_step(f"hybrid_search_qdrant() sparse vector ready, nnz={len(sparse_indices)}")

    log_step(
        f"Hybrid query_points on collection={collection_name} "
        f"(dense_limit={dense_limit}, sparse_limit={sparse_limit}, final_top_k={final_top_k})"
    )

    try:
        res = qdrant.query_points(
            collection_name=collection_name,
            prefetch=[
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="bm25",
                    limit=sparse_limit,
                ),
                qm.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=dense_limit,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=final_top_k,
            with_payload=True,
            with_vectors=False,
        )
        log_step("Qdrant query_points() call succeeded")
    except Exception as e:
        log_exception("Qdrant query_points() failed", e)
        raise

    points = getattr(res, "points", [])
    log_step(f"Hybrid search returned {len(points)} points.")
    if points:
        first = points[0]
        log_step(
            f"First point: id={first.id}, score={first.score}, "
            f"payload_keys={list((first.payload or {}).keys())}"
        )

    return query, collection_name, points


# ============================================================
# ========= CONTEXT BUILDING + CANDIDATE CITATIONS ===========
# ============================================================

def build_context_and_citations(
    points,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> Tuple[str, List[str]]:
    """
    Build text context for LLM and collect unique citation links for emails.

    IMPORTANT:
    - Matches ETL payload keys:
      * primary:   `chunk_text`        (plain text body chunk)
      * fallback:  `chunk_text_plain`  (legacy plain text key, if present)
    - We no longer use any HTML fields here.
    """
    log_step(
        f"build_context_and_citations() called with {len(points)} points, "
        f"max_chars={max_chars}"
    )
    context_parts: List[str] = []
    total_len = 0
    citations: List[str] = []

    for idx, p in enumerate(points):
        payload = p.payload or {}
        log_step(
            f"Processing point[{idx}] id={p.id}, score={p.score}, "
            f"payload_keys={list(payload.keys())}"
        )

          # Align with ETL payload:
        #   - primary key:  "chunk_text"        (current ETL, plain text)
        #   - fallback key: "chunk_text_plain"  (legacy plain text, if ever present)
        chunk_text = (
            payload.get("chunk_text")          # primary (new ETL)
            or payload.get("chunk_text_plain") # legacy fallback
            or ""
        )

        if not chunk_text.strip():
            log_step(f"Skipping point[{idx}] because chunk_text is empty")
            continue

        subject = payload.get("subject", "")
        sender_name = payload.get("sender_name", "")
        sender_address = payload.get("sender_address", "")
        received = payload.get("received_datetime", "")
        importance = payload.get("importance_level", "unknown")
        category = payload.get("category", "unknown")
        main_topic = payload.get("main_topic") or ""
        action_required = payload.get("action_required", "none")

        source_link = (
            payload.get("source")
            or payload.get("web_link")
            or f"email_id:{payload.get('email_id', 'unknown')}"
        )

        if source_link and source_link not in citations:
            citations.append(source_link)
            log_step(f"Added citation: {source_link}")

        if main_topic:
            summary = main_topic
        elif subject:
            summary = subject
        else:
            summary = chunk_text[:200]

        header = (
            f"Email: {subject}\n"
            f"From: {sender_name} <{sender_address}>\n"
            f"Received: {received}\n"
            f"Importance: {importance} | Category: {category} | Action: {action_required}\n"
            f"Source: {source_link}\n"
        )

        chunk = (
            f"{header}"
            f"Summary: {summary}\n"
            f"Content:\n{chunk_text}\n"
            f"{'-' * 80}\n"
        )

        if total_len + len(chunk) > max_chars:
            log_step(
                f"Reached max_chars limit: current={total_len}, "
                f"next_chunk_len={len(chunk)}, limit={max_chars}. Stopping."
            )
            break

        context_parts.append(chunk)
        total_len += len(chunk)
        log_step(
            f"Added point[{idx}] to context, chunk_len={len(chunk)}, "
            f"total_len={total_len}"
        )

    context = "".join(context_parts)
    log_step(
        f"Built context of length {len(context)} chars from {len(context_parts)} chunks; "
        f"{len(citations)} unique citation links."
    )
    return context, citations


# ============================================================
# ================= RAG ANSWERING (CORE) =====================
# ============================================================

def answer_with_rag(
    user_query: str,
    user_email: str,
    history: Optional[List["ChatMessage"]] = None,
) -> dict:
    """
    Main pipeline:

    1) Classify query using current question + history.
    2) If needs_email_rag == False → pure chat (no Qdrant).
    3) If needs_email_rag == True → email RAG flow against Qdrant.
    """
    log_step(
        f"[DEBUG] answer_with_rag() called with user_query={user_query!r}, "
        f"user_email={user_email!r}, history_len={len(history) if history else 0}"
    )

    classification = classify_query_with_history(user_query, history)
    query_type = classification["query_type"]
    needs_email_rag = classification["needs_email_rag"]

    log_step(
        f"[DEBUG] answer_with_rag(): query classification -> "
        f"type={query_type}, needs_email_rag={needs_email_rag}"
    )

    # CASE 1: NO EMAIL RAG NEEDED
    if not needs_email_rag:
        log_step(
            "[DEBUG] answer_with_rag(): classifier says NO email RAG needed → pure chat."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, helpful assistant. "
                    "Answer based on general knowledge, the conversation history, "
                    "and any email content that is already shown in the chat history. "
                    "Do NOT mention Qdrant or retrieval internals. "
                    "Just help the user naturally."
                ),
            },
        ]

        if history:
            log_step(
                "answer_with_rag(): adding history messages to pure-chat LLM call"
            )
            for m in history:
                messages.append({"role": m.role, "content": m.content})

        messages.append({"role": "user", "content": user_query})

        answer = _llm_chat(messages, temperature=0.5, max_tokens=1024).strip()
        log_step("answer_with_rag(): pure-chat LLM call completed")

        return {
            "question": user_query,
            "user_email": user_email,
            "collection": None,
            "answer": answer,
            "citations": [],
        }

    # CASE 2: EMAIL RAG NEEDED
    log_step(
        "[DEBUG] answer_with_rag(): classifier says EMAIL RAG needed → "
        "entering email-based RAG flow"
    )

    try:
        query, collection_name, points = hybrid_search_qdrant(user_query, user_email)
        log_step(
            f"answer_with_rag(): hybrid_search_qdrant completed, "
            f"resolved_query={query!r}, collection_name={collection_name}, "
            f"num_points={len(points)}"
        )
    except Exception as e:
        log_exception("answer_with_rag(): hybrid_search_qdrant raised exception", e)
        raise

    if not points:
        log_step(
            "No Qdrant results for this mailbox. Friendly fallback answer (no citations)."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly assistant. For this request, no relevant email data "
                    "was found for the user's mailbox. Be honest about that, but still:\n"
                    "- Greet the user politely if appropriate.\n"
                    "- Answer any general part of the question from your own knowledge.\n"
                    "- Suggest how they might try a different query, e.g. by mentioning "
                    "keywords that could appear in their emails (project name, client, etc.).\n"
                    "- DO NOT add any email citations."
                ),
            },
        ]

        if history:
            log_step(
                "answer_with_rag(): adding history messages to fallback LLM call"
            )
            for m in history:
                messages.append({"role": m.role, "content": m.content})

        messages.append({"role": "user", "content": query})

        answer = _llm_chat(messages, temperature=0.3, max_tokens=1024).strip()
        log_step("answer_with_rag(): fallback LLM call completed")

        return {
            "question": query,
            "user_email": user_email,
            "collection": collection_name,
            "answer": answer,
            "citations": [],
        }

    context, citations = build_context_and_citations(points)
    log_step(
        f"answer_with_rag(): context built, length={len(context)}, "
        f"num_citations={len(citations)}"
    )

    system_prompt = """
You are a polite, helpful assistant that can do two different things:

1) Chat naturally with the user (greetings, small talk, general questions)
2) Answer questions based on retrieved email context

You will receive:
- The user's question
- A block of email context (may be empty, partial, or full)
- A list of candidate email sources (URLs or IDs)

========================
HOW TO DECIDE YOUR BEHAVIOUR
========================

A) PURE SMALL TALK / GENERAL CHAT
- If the user is just greeting (e.g. "hi", "hello", "hey", "how are you"),
  or asking something that clearly does NOT need email data
  (e.g. "how to write a polite email", "how is your day"),
  then:
  - Answer like a normal friendly assistant.
  - DO NOT mention emails.
  - DO NOT mention context.
  - DO NOT show any citations.

B) EMAIL-BASED QUESTIONS
- If the question is about something that can be answered using emails
  (projects, clients, invoices, tasks, meetings, etc.), then:
  - Carefully read the email context.
  - Use ONLY the information that is actually present in that context
    when making claims about specific emails.
  - Be clear, concise, and structured (bullet points are OK).

========================
CITATIONS RULES (VERY IMPORTANT)
========================

You are also given a list of candidate email sources.

1) Only show citations if you actually used information from one or more email chunks.
   - "Use" means you relied on a specific email's content or metadata
     to support your answer (subject, sender, dates, actions, etc.).

2) If you answer purely from general knowledge or small talk
   (for example, when greeting the user or giving generic advice),
   then:
   - DO NOT show any citations.
   - DO NOT invent fake citations.

3) If you partially used email context and partially used general knowledge:
   - You may still show citations, BUT only for the specific emails you truly used.

4) Citation format when you DO use email information:
   - END your answer with a section exactly like:

     Citations:
     - <source-1>
     - <source-2>

   - If you did NOT use any email context at all,
     completely omit the "Citations:" section.

========================
WHEN CONTEXT IS NOT ENOUGH
========================

- If the email context does not contain enough information
  to fully answer an email-related question:
  1) Say clearly that the retrieved emails do not provide enough detail.
  2) Optionally give general advice or suggestions from your own knowledge,
     but do NOT pretend that advice came from the emails.
  3) In that case:
     - If you did NOT use any email content, do NOT show citations.
     - If you did use at least one email partially, only cite those you used.

========================
SUMMARY
========================

- Be warm and conversational.
- Use email context only when it truly helps.
- Only show citations when you actually relied on one or more email chunks.
- No citations for pure greetings or purely general answers.
"""

    sources_block = ""
    if citations:
        sources_block = "Candidate email sources you may cite IF you use them:\n"
        sources_block += "\n".join(f"- {c}" for c in citations)
        sources_block += "\n\n"
        log_step(
            f"answer_with_rag(): sources_block prepared with {len(citations)} entries"
        )

    user_content = (
        f"User email: {user_email}\n"
        f"User question:\n{query}\n\n"
        f"--- Retrieved email context start ---\n"
        f"{context}\n"
        f"--- Retrieved email context end ---\n\n"
        f"{sources_block}"
        f"Now answer the user's question following ALL the rules above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if history:
        log_step(
            "answer_with_rag(): adding history messages to main RAG LLM call"
        )
        for m in history[-20:]:
            messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": user_content})

    log_step("Calling LLM for final RAG answer (email-based)...")
    answer = _llm_chat(messages, temperature=0.1, max_tokens=16384).strip()
    log_step("answer_with_rag(): LLM call for RAG answer completed")

    return {
        "question": query,
        "user_email": user_email,
        "collection": collection_name,
        "answer": answer,
        "citations": citations,
    }


# ============================================================
# ====================== FASTAPI LAYER =======================
# ============================================================

app = FastAPI(title="Email RAG Query API", version="1.0.0")


@app.middleware("http")
async def log_raw_request(request: Request, call_next):
    try:
        body_bytes = await request.body()
        try:
            parsed = json.loads(body_bytes or b"{}")
            log_step(f"[RAW REQUEST] {parsed}")
        except Exception:
            log_step(
                "[RAW REQUEST] (non-JSON) "
                + body_bytes.decode("utf-8", errors="ignore")
            )
    except Exception as e:
        log_exception("ERROR READING RAW REQUEST", e)

    response = await call_next(request)
    return response


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class QueryRequest(BaseModel):
    question: str
    user_email: str
    # n8n is sending history as a JSON STRING, so accept str here
    history: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    user_email: str
    collection: Optional[str]
    answer: str
    citations: List[str]


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    log_step(
        f"Received /query request from client="
        f"{request.client.host if request.client else 'unknown'}: "
        f"user_email={body.user_email!r} | question={body.question!r}"
    )

    # ===================== PARSE HISTORY =====================
    raw_history = body.history
    history_list: Optional[List[ChatMessage]] = None

    if raw_history is None or raw_history == "":
        log_step("[DEBUG] No history provided or empty string from client")
    else:
        log_step(f"[DEBUG] Raw history string from client: {raw_history!r}")
        try:
            parsed = json.loads(raw_history)
            if isinstance(parsed, list):
                history_list = [ChatMessage(**msg) for msg in parsed]
                log_step(
                    f"[DEBUG] Parsed history into list, len={len(history_list)}"
                )
            else:
                log_step(
                    "[DEBUG] Parsed history JSON is not a list. Ignoring."
                )
        except Exception as e:
            log_exception(
                "[DEBUG] Failed to parse history string as JSON", e
            )

    if history_list:
        log_step(
            f"[DEBUG] Normalized chat history passed to RAG, len={len(history_list)}"
        )
    else:
        log_step("[DEBUG] No usable history for RAG")

    # ========================================================

    try:
        log_step("query_endpoint(): calling answer_with_rag()")
        result = answer_with_rag(body.question, body.user_email, history_list)
        log_step(
            "query_endpoint(): answer_with_rag() completed successfully "
            f"collection={result.get('collection')}, "
            f"answer_len={len(result.get('answer', ''))}, "
            f"citations_count={len(result.get('citations', []))}"
        )
        response = QueryResponse(**result)
        log_step("query_endpoint(): returning successful response to client")
        return response

    except ValueError as e:
        log_exception("query_endpoint(): ValueError (400 Bad Request)", e)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        log_exception("query_endpoint(): RuntimeError (404 Not Found)", e)
        raise HTTPException(status_code=404, detail=str(e))
    except requests.exceptions.RequestException as e:
        log_exception(
            "query_endpoint(): Downstream HTTP error (LLM or Qdrant)", e
        )
        raise HTTPException(
            status_code=502, detail="Downstream service error (LLM or Qdrant)."
        )
    except Exception as e:
        log_exception(
            "query_endpoint(): Unexpected error (500 Internal Server Error)", e
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================
# ===================== CLI ENTRY (OPTIONAL) =================
# ============================================================

if __name__ == "__main__":
    import sys

    log_step("[CLI] Script started with arguments: " + " ".join(sys.argv[1:]))

    if len(sys.argv) < 3:
        print(
            "Usage: python query_terminal_rag_fast_api.py "
            "\"your question here\" user_email@example.com"
        )
        print(
            "Or run as API server with:\n  "
            "uvicorn query_terminal_rag_fast_api:app --host 0.0.0.0 --port 8000"
        )
        sys.exit(1)

    raw_q = sys.argv[1]
    user_email_arg = sys.argv[2]

    log_step(
        f"[CLI] Running single query for user={user_email_arg!r}, "
        f"question={raw_q!r}"
    )
    try:
        result = answer_with_rag(raw_q, user_email_arg)
    except Exception as e:
        log_exception("[CLI] Fatal error while answering", e)
        raise

    print("\n--- Query (used as-is) ---")
    print(result["question"])
    print("User:", result["user_email"])
    print("Collection:", result["collection"])
    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Candidate citations (for debugging) ---")
    for c in result["citations"]:
        print("-", c)

    log_step("[CLI] Done.")
