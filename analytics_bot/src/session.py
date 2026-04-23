"""
session.py
==========
Module-level session state: caches, conversation history, charts, last results.
All state is intentionally module-global so every import shares the same instance.
"""
from __future__ import annotations
import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# ── Session state globals ──────────────────────────────────────
_query_cache: dict = {}
_semantic_index: list = []               # list of {key, embedding, use_viz, use_reco}
_conversation_history: list = []          # list of dicts: {question, rewritten, columns, success}
_accumulated_recommendations: list = []   # list of recommendation strings
_query_history: list = []                 # list of dicts: {question, status, attempts, timestamp}
_session_charts: list = []               # list of file paths to temp PNGs
_last_result: Optional[pd.DataFrame] = None
_last_fig: Optional[go.Figure] = None
_last_plotly_code: str = ""   # stored for chart-edit feature

_SEMANTIC_THRESHOLD = 0.70
_SEMANTIC_MAX_ENTRIES = 500


def _detect_lang(text: str) -> str:
    """Return 'ar' if text is predominantly Arabic, else 'en'."""
    arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
    return 'ar' if arabic / max(len(text), 1) > 0.15 else 'en'

MAX_HISTORY_TURNS = 6  # keep N most recent turns for context


# ── Cache helpers ──────────────────────────────────────────────
def clear_cache() -> None:
    """Wipe result cache and semantic index."""
    _query_cache.clear()
    _semantic_index.clear()


def _semantic_lookup(
    question: str,
    use_viz: bool,
    use_reco: bool,
    lang: str = "en",
    top_n=None,
    time_filter=None,
    dimension: str = "general",
):
    """
    Return a cached result dict if a semantically similar question was already
    answered with matching flags, top_n, and time_filter.
    Returns None on any failure.
    """
    if not _semantic_index:
        return None
    try:
        import numpy as np
        from analytics_bot.src.config import embeddings
        if embeddings is None:
            return None
        q_emb = np.array(embeddings.embed_query(question), dtype="float32")
        norm = np.linalg.norm(q_emb)
        if norm == 0:
            return None
        q_emb = q_emb / norm

        best_score, best_key = -1.0, None
        for entry in _semantic_index:
            if entry["use_viz"] != use_viz or entry["use_reco"] != use_reco:
                continue
            if entry.get("lang", "en") != lang:
                continue
            # Exact match on top_n: if both sides have a value they must agree
            if top_n is not None and entry.get("top_n") is not None:
                if top_n != entry["top_n"]:
                    continue
            # Exact match on time_filter
            if time_filter is not None and entry.get("time_filter") is not None:
                if str(time_filter) != str(entry["time_filter"]):
                    continue
            # Exact match on dimension (LLM-extracted business entity)
            if dimension != "general" and entry.get("dimension", "general") != "general":
                if dimension != entry["dimension"]:
                    continue
            sim = float(np.dot(q_emb, entry["embedding"]))
            if sim > best_score:
                best_score, best_key = sim, entry["key"]

        print(f"🔍 Semantic score: {best_score:.4f} (threshold={_SEMANTIC_THRESHOLD}, lang={lang}, top_n={top_n}, time={time_filter}, dim={dimension}, index={len(_semantic_index)})")
        if best_score >= _SEMANTIC_THRESHOLD and best_key and best_key in _query_cache:
            return _query_cache[best_key]
    except Exception as e:
        print(f"⚠️ Semantic lookup error: {e}")
    return None


def _semantic_store(
    question: str,
    cache_key: str,
    use_viz: bool,
    use_reco: bool,
    lang: str = "en",
    top_n=None,
    time_filter=None,
    dimension: str = "general",
) -> None:
    """Embed question and append to the semantic index (pointer to cache_key only)."""
    try:
        import numpy as np
        from analytics_bot.src.config import embeddings
        if embeddings is None:
            return
        emb = np.array(embeddings.embed_query(question), dtype="float32")
        norm = np.linalg.norm(emb)
        if norm == 0:
            return
        emb = emb / norm
        _semantic_index.append({
            "key":         cache_key,
            "embedding":   emb,
            "use_viz":     use_viz,
            "use_reco":    use_reco,
            "lang":        lang,
            "top_n":       top_n,
            "time_filter": str(time_filter) if time_filter is not None else None,
            "dimension":   dimension,
        })
        if len(_semantic_index) > _SEMANTIC_MAX_ENTRIES:
            del _semantic_index[0]
    except Exception:
        pass


# ── Full session reset ─────────────────────────────────────────
def clear_memory() -> None:
    """Reset all session state and remove temp chart files."""
    global _last_result, _last_fig

    _query_cache.clear()
    _semantic_index.clear()
    _conversation_history.clear()
    _accumulated_recommendations.clear()
    _query_history.clear()

    # Remove temp PNG files
    for path in _session_charts:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    _session_charts.clear()

    _last_result = None
    _last_fig = None
    global _last_plotly_code
    _last_plotly_code = ""

    # Reset token tracker
    try:
        from src.llm import _token_tracker
        _token_tracker.reset()
    except Exception:
        pass


# ── History context builder ────────────────────────────────────
def _build_history_context() -> str:
    """Return recent conversation history as a formatted string for prompts."""
    if not _conversation_history:
        return "No previous queries in this session."

    recent = _conversation_history[-MAX_HISTORY_TURNS:]
    lines = []
    for i, entry in enumerate(recent, 1):
        lines.append(
            f"[{i}] Q: {entry.get('question', '')}\n"
            f"    Rewritten: {entry.get('rewritten', '')}\n"
            f"    Columns returned: {entry.get('columns', [])}\n"
            f"    Success: {entry.get('success', False)}"
        )
    return "\n".join(lines)


def _add_to_history(
    question: str,
    rewritten: str,
    columns: list,
    success: bool,
) -> None:
    """Append a query/result summary to conversation history."""
    _conversation_history.append(
        {
            "question": question,
            "rewritten": rewritten,
            "columns": columns,
            "success": success,
        }
    )
    # Trim to 2× MAX_HISTORY_TURNS to avoid unbounded growth
    if len(_conversation_history) > MAX_HISTORY_TURNS * 2:
        del _conversation_history[: len(_conversation_history) - MAX_HISTORY_TURNS]


def _add_query_to_history(question: str, status: str, shape: str) -> None:
    from datetime import datetime
    _query_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "question": question,
        "status": status,
        "shape": shape
    })
    if len(_query_history) > 30:
        del _query_history[0]
# ── Recommendations context ────────────────────────────────────
def _build_recommendations_context() -> str:
    """Return a compact summary of recommendations for use in LLM prompts."""
    if not _accumulated_recommendations:
        return "No recommendations generated yet."
    lines = []
    for i, r in enumerate(_accumulated_recommendations):
        # Use truncated form for the LLM context to avoid token bloat
        if isinstance(r, dict):
            q   = r.get("question", "")[:80]
            rec = r.get("recommendation", "")[:200]
            lines.append(f"[Reco {i+1}] Q: {q}\n{rec}")
        else:
            lines.append(f"[Reco {i+1}] {str(r)[:200]}")
    return "\n\n".join(lines)


def _add_recommendation_to_memory(
    question: str,
    recommendation: str,
    chart_path: str | None = None,
) -> None:
    """Store the full recommendation as a dict for PDF export and cross-query context."""
    _accumulated_recommendations.append({
        "question": question,
        "recommendation": recommendation,
        "chart_path": chart_path,
    })
    # Keep last 10 to avoid unbounded growth
    while len(_accumulated_recommendations) > 10:
        del _accumulated_recommendations[0]


# ── Session Persistence ────────────────────────────────────────
def save_session(path: str) -> str:
    """Serialize conversation history, recommendations, and query history to JSON."""
    import json
    data = {
        "conversation_history":       _conversation_history,
        "accumulated_recommendations": _accumulated_recommendations,
        "query_history":               _query_history,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        n_turns = len(_conversation_history)
        n_recos = len(_accumulated_recommendations)
        return f"✅ Session saved → {path}\n   {n_turns} turns · {n_recos} recommendations stored."
    except Exception as e:
        return f"❌ Save failed: {e}"


def load_session(path: str) -> str:
    """Restore conversation history, recommendations, and query history from JSON."""
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _conversation_history.clear()
        _conversation_history.extend(data.get("conversation_history", []))
        _accumulated_recommendations.clear()
        _accumulated_recommendations.extend(data.get("accumulated_recommendations", []))
        _query_history.clear()
        _query_history.extend(data.get("query_history", []))
        n_turns = len(_conversation_history)
        n_recos = len(_accumulated_recommendations)
        return (
            f"✅ Session loaded from {path}\n"
            f"   {n_turns} conversation turns · {n_recos} recommendations restored."
        )
    except FileNotFoundError:
        return f"❌ No saved session found at {path}"
    except Exception as e:
        return f"❌ Load failed: {e}"
