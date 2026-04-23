"""
intent.py
=========
LLM-based intent classification, query rewriting, chitchat detection,
reference resolution, and spelling correction.
"""
from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser

from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import (
    CHITCHAT_GATE_PROMPT,
    CHITCHAT_RESPONSE_PROMPT,
    DECOMPOSE_PROMPT,
    INTENT_PROMPT,
    REWRITE_PROMPT,
    REFERENCE_RESOLVE_PROMPT,
    SPELL_CORRECT_PROMPT,
    CHART_EDIT_GATE_PROMPT,
    CHART_EDIT_PROMPT,
)
from analytics_bot.src.session import _conversation_history, _build_history_context


# ── Spelling / typo correction ─────────────────────────────────
def _correct_spelling(question: str) -> str:
    """LLM pre-pass to fix obvious Arabic/English typos before any other step."""
    try:
        corrected = (SPELL_CORRECT_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        ).strip()
        # Sanity guard: don't return empty or suspiciously long answer
        if corrected and len(corrected) < len(question) * 3:
            return corrected
        return question
    except Exception:
        return question


# ── Reference resolver ─────────────────────────────────────────
def _resolve_references(question: str) -> str:
    """Expand implicit pronouns/references or short follow-up fragments using conversation history."""
    if not _conversation_history:
        return question
    ref_words = [
        "same", "them", "it",
        "نفس", "هم", "هو", "هي", "ذلك", "تلك", "ذات",
    ]
    # Also trigger for short follow-up fragments (e.g. "والان خلال 2024 فقط")
    is_short_followup = len(question.strip()) < 40 or question.strip().startswith("و")
    if not (any(w in question for w in ref_words) or is_short_followup):
        return question
    try:
        resolved = (REFERENCE_RESOLVE_PROMPT | llm | StrOutputParser()).invoke({
            "question": question,
            "history":  _build_history_context(),
        }).strip()
        return resolved or question
    except Exception:
        return question


# ── Chitchat gate ──────────────────────────────────────────────
def _is_analytics_query(question: str, history_context: str = "") -> bool:
    """Return True if the question is a data/analytics query; False if chitchat."""
    try:
        label = (CHITCHAT_GATE_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question, "history": history_context or "No previous queries."}
        ).strip().lower()
        if label.startswith("analytics"):
            return True
        if label.startswith("chitchat"):
            return False
        return True   # default to analytics on ambiguous response
    except Exception:
        return True


def _get_chitchat_response(question: str) -> str:
    """Generate a friendly chitchat response."""
    try:
        return (CHITCHAT_RESPONSE_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        ).strip()
    except Exception as e:
        return f"مرحباً! أنا مساعد تحليل البيانات. ({e})"


# ── Query rewriter ─────────────────────────────────────────────
def _rewrite_query(question: str, history_context: str = "") -> str:
    """Rewrite user question into a precise analytical intent in English."""
    return (REWRITE_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "history": history_context or "No previous queries."}
    ).strip()


# ── Intent classifier ──────────────────────────────────────────
def _classify_intent(question: str) -> dict:
    """Return a dict: {intent_type, chart_type, needs_chart, top_n, time_filter}."""
    defaults = {
        "intent_type": "detail",
        "chart_type": "vbar",
        "needs_chart": True,
        "top_n": None,
        "time_filter": None,
        "dimension": "general",
    }
    try:
        raw = (INTENT_PROMPT | llm | StrOutputParser()).invoke({"question": question}).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        # Extract JSON object if there is surrounding text
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        intent = json.loads(raw)
        for k, v in defaults.items():
            intent.setdefault(k, v)
        # Null-guard critical fields the LLM may return as null
        if not intent.get("chart_type"):
            intent["chart_type"] = "vbar"
        if not intent.get("dimension"):
            intent["dimension"] = "general"
        # Validate intent_type
        valid_intents = {"ranking", "trend", "distribution", "comparison", "correlation", "detail"}
        if intent.get("intent_type") not in valid_intents:
            intent["intent_type"] = "detail"
        return intent
    except Exception:
        return defaults


# ── Query decomposer ───────────────────────────────────────────
def _decompose_query(question: str) -> dict:
    """
    Return {is_compound, steps, combination}.
    Falls back gracefully on any LLM/parse failure.
    """
    fallback = {
        "is_compound": False,
        "steps": [question],
        "combination": "display_separately",
    }
    try:
        raw = (DECOMPOSE_PROMPT | llm | StrOutputParser()).invoke({"question": question}).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        plan = json.loads(raw)
        plan.setdefault("is_compound", False)
        plan.setdefault("steps", [question])
        plan.setdefault("combination", "display_separately")
        return plan
    except Exception:
        return fallback


# ── Chart Edit helpers ────────────────────────────────────────
def _is_chart_edit(question: str) -> bool:
    """
    Return True if the user wants to modify the currently displayed chart
    (change type, flip orientation, recolor, sort differently, etc.).
    Returns False on any error — safer to fall through to normal pipeline.
    """
    try:
        label = (CHART_EDIT_GATE_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question}
        ).strip().lower()
        return label.startswith("chart_edit")
    except Exception:
        return False


def _apply_chart_edit(instruction: str, plotly_code: str, result) -> str:
    """
    Rewrite existing plotly code to apply the user's chart modification instruction.
    Returns the original code unchanged on any LLM error.
    """
    try:
        preview = result.head(15).to_string(index=False) if result is not None else ""
        cols    = list(result.columns) if result is not None else []
        return (CHART_EDIT_PROMPT | llm | StrOutputParser()).invoke({
            "instruction":  instruction,
            "plotly_code":  plotly_code,
            "columns":      cols,
            "data_preview": preview,
        }).strip()
    except Exception:
        return plotly_code  # fallback: keep existing chart
