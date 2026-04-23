"""End-to-end smoke test: ask_retail_rag_ui() against the DWH via SQL path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics_bot.src.pipeline import ask_retail_rag_ui  # noqa: E402


QUESTIONS = [
    "What are the top 5 brands by revenue?",
    "Show monthly revenue trend in 2024",
]

for q in QUESTIONS:
    print("\n" + "=" * 80)
    print(f"Q: {q}")
    print("=" * 80)
    out = ask_retail_rag_ui(q, use_viz=False, use_reco=False, use_cache=False, max_retries=3)
    print("LOG:")
    print(out["log"])
    print("\nCHAT:")
    print(out["chat_text"][:800])
    print(f"\nTokens: {out['tokens_used']:,}")
