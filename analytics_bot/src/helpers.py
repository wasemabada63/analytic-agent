"""
helpers.py
==========
Small, dependency-free utility functions.
"""
import hashlib
import re


def _strip_fences(raw: str) -> str:
    """Remove ```python ... ``` (or plain ```) fences from LLM code output."""
    raw = raw.strip()
    # Remove opening fence (```python or ```)
    raw = re.sub(r"^```(?:python)?\n?", "", raw)
    # Remove closing fence
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _get_cache_key(question: str, use_viz: bool = True, use_reco: bool = True) -> str:
    """Return a stable SHA-256 hex digest for a question string and options."""
    key_str = f"{question.strip().lower()}|{use_viz}|{use_reco}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()
