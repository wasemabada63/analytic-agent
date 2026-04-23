"""
llm.py
======
LLM singleton (ChatGroq) and the token-tracking callback.
All other modules import `llm` and `_token_tracker` from here.
"""
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from analytics_bot.src.config import GROQ_API_KEY


# ── Token-tracking callback ───────────────────────────────────
class _TokenTracker(BaseCallbackHandler):
    """Accumulates prompt + completion token counts across all LLM calls."""

    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0

    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                # Try generation_info.usage first (preferred source)
                meta  = getattr(g, "generation_info", None) or {}
                usage = meta.get("usage", {})
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)

                # Fall back to response_metadata only when generation_info gave nothing.
                # Using OR logic avoids double-counting when Groq populates both fields.
                if pt == 0 and ct == 0 and hasattr(g, "message"):
                    rm = getattr(g.message, "response_metadata", {}) or {}
                    tu = rm.get("token_usage", {})
                    pt = tu.get("prompt_tokens", 0)
                    ct = tu.get("completion_tokens", 0)

                self.prompt_tokens     += pt
                self.completion_tokens += ct

    def reset(self):
        self.prompt_tokens = self.completion_tokens = 0


# ── Singletons ────────────────────────────────────────────────
_token_tracker = _TokenTracker()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0,
    callbacks=[_token_tracker],
)
print("✅ LLM ready.\n")
