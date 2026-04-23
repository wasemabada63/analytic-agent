"""
executor.py
===========
Secure code execution, schema retrieval, pandas loop, combine step.
Also: SQL validation and execution against the DWH (dwh1 star schema).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text

from analytics_bot.src.config import (
    orders, products, order_items, categories, subcategories, vector_store,
    engine, DWH_SCHEMA,
)
from analytics_bot.src.helpers import _strip_fences
from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import (
    COMBINE_PROMPT,
    SQL_PROMPT, SQL_FIX_PROMPT,
)
from analytics_bot.utils.arabic import fix_arabic

# ── Security: forbidden patterns ──────────────────────────────
_FORBIDDEN = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\b__import__\s*\(",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bshutil\b",
    r"os\.system",
    r"os\.popen",
    r"os\.remove",
]


def _validate_code_security(code: str) -> None:
    for pattern in _FORBIDDEN:
        if re.search(pattern, code):
            raise ValueError(f"\U0001f6ab Security violation: '{pattern}'")


# ── Sanity check result ────────────────────────────────────────
def _sanity_check_result(result: pd.DataFrame, intent: dict) -> Tuple[bool, str]:
    if result is None or result.empty:
        return False, "\u26a0\ufe0f Result is empty."
    nums = result.select_dtypes(include="number").columns.tolist()
    if nums and result[nums].abs().sum().sum() == 0:
        return False, "\u26a0\ufe0f All numeric values are 0 \u2014 likely missing a JOIN or wrong filter."
    if intent.get("intent_type") == "ranking" and intent.get("top_n") and len(result) == 1:
        return False, f"\u26a0\ufe0f Ranking query expected {intent['top_n']} rows but got 1."
    if intent.get("intent_type") == "trend":
        if not any(
            kw in c.lower()
            for c in result.columns
            for kw in ["month", "date", "year", "week", "day"]
        ):
            return False, "\u26a0\ufe0f Trend query has no time column."
    return True, ""


# ── Schema context retrieval ───────────────────────────────────
def _build_schema_context(question: str, top_k: int = 5) -> str:
    docs = vector_store.similarity_search_with_score(question, k=top_k * 2)
    docs.sort(key=lambda x: x[1])
    parts = []
    for d, _ in docs[:top_k]:
        m = d.metadata
        parts.append(
            f"Table: {m.get('table_name', '')}\n"
            f"Description: {m.get('description', '')}\n"
            f"Columns: {', '.join(m.get('columns', []))}\n"
            f"Types: {m.get('column_types', {})}\n"
            f"---\n{d.page_content}"
        )
    return "\n\n".join(parts)


# ── Safe code execution ────────────────────────────────────────
def _exec_code(code: str, extra_ns: dict) -> dict:
    _validate_code_security(code)
    local_ns = {
        "pd": pd,
        "px": px,
        "go": go,
        "fix_arabic": fix_arabic,
        "orders": orders,
        "products": products,
        "order_items": order_items,
        "categories": categories,
        "subcategories": subcategories,
        "sub_categories": subcategories,   # alias for robustness
        **extra_ns,
    }
    exec(code, local_ns)
    return local_ns


# ══════════════════════════════════════════════════════════════
# SQL path — validation + execution against dwh1
# ══════════════════════════════════════════════════════════════

_SQL_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|"
    r"COPY|MERGE|CALL|DO|VACUUM|ANALYZE|CLUSTER|REINDEX)\b",
    re.IGNORECASE,
)
_SQL_COMMENT = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)
_MAX_ROWS = 10_000


def _strip_sql(sql: str) -> str:
    """Remove markdown fences, leading/trailing whitespace and wrapping backticks."""
    s = _strip_fences(sql).strip()
    # sometimes the LLM wraps in ```sql blocks or a single backtick
    s = s.strip("`").strip()
    # drop trailing semicolon for parsing; re-add once at the end
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _validate_sql(sql: str) -> str:
    """
    Return the sanitized SQL or raise ValueError.
    - Must be a single SELECT statement.
    - No DDL / DML.
    - Forces a LIMIT if missing (cap at _MAX_ROWS).
    """
    clean = _strip_sql(sql)
    if not clean:
        raise ValueError("Empty SQL.")

    # Strip comments before forbidden-keyword check so they aren't false positives.
    no_comments = _SQL_COMMENT.sub(" ", clean)

    if ";" in no_comments:
        raise ValueError("Multiple statements are not allowed.")

    if not re.match(r"^\s*(WITH|SELECT)\b", no_comments, re.IGNORECASE):
        raise ValueError("Only SELECT (optionally prefixed with WITH) is allowed.")

    if _SQL_FORBIDDEN.search(no_comments):
        raise ValueError("Forbidden SQL keyword detected (DDL/DML blocked).")

    # Enforce a LIMIT at the outer level. If one is missing, tack one on.
    if not re.search(r"\bLIMIT\s+\d+\s*$", no_comments, re.IGNORECASE):
        clean = f"{clean}\nLIMIT {_MAX_ROWS}"
    else:
        # cap user-supplied LIMIT
        def _cap(m):
            n = min(int(m.group(1)), _MAX_ROWS)
            return f"LIMIT {n}"
        clean = re.sub(r"\bLIMIT\s+(\d+)\s*$", _cap, clean, flags=re.IGNORECASE)

    return clean + ";"


def _exec_sql(sql: str) -> pd.DataFrame:
    """
    Validate and execute a SELECT against the DWH engine.
    Returns the result as a DataFrame.
    """
    safe_sql = _validate_sql(sql)
    with engine.connect() as conn:
        # Connection-level read-only is redundant with prompt rules + validator,
        # but cheap defense-in-depth:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        df = pd.read_sql_query(text(safe_sql), conn)
    return df


# ── SQL step runner (mirror of _run_pandas_step) ───────────────
def _run_sql_step(
    step_question: str,
    schema_context: str,
    intent_hint: str,
    history_context: str,
    max_retries: int,
    step_num: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Returns (DataFrame, final_sql) — or (None, last_sql_attempt) on failure.
    """
    sql: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            sql = _strip_sql(
                (SQL_PROMPT | llm | StrOutputParser()).invoke({
                    "schema_context":  schema_context,
                    "question":        step_question,
                    "intent_hint":     intent_hint,
                    "history_context": history_context,
                })
            )
        else:
            sql = _strip_sql(
                (SQL_FIX_PROMPT | llm | StrOutputParser()).invoke({
                    "sql":            sql,
                    "error":          str(last_error),
                    "question":       step_question,
                    "schema_context": schema_context,
                })
            )
        try:
            df = _exec_sql(sql)
            return df, sql
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                return None, sql

    return None, sql


# ── Combine sub-step results ───────────────────────────────────
def _combine_step_results(
    question: str,
    steps: list,
    combination: str,
    step_results: list,
    max_retries: int,
):
    if len(step_results) == 1:
        return step_results[0]

    # display_separately: return the list — pipeline renders each table independently
    if combination == "display_separately":
        return step_results

    info_parts = [
        f"step_result_{i} (shape {df.shape}):\n"
        f"  columns: {list(df.columns)}\n"
        f"  preview:\n{df.head(5).to_string(index=False)}"
        for i, df in enumerate(step_results)
    ]
    extra_ns = {f"step_result_{i}": df for i, df in enumerate(step_results)}

    combine_code: Optional[str] = None
    last_error = None

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            combine_code = _strip_fences(
                (COMBINE_PROMPT | llm | StrOutputParser()).invoke({
                    "question": question,
                    "steps": "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)),
                    "combination": combination,
                    "step_results_info": "\n\n".join(info_parts),
                })
            )
        else:
            combine_code = _strip_fences(
                (PANDAS_FIX_PROMPT | llm | StrOutputParser()).invoke({
                    "code": combine_code,
                    "error": str(last_error),
                    "question": f"Combine: {question}",
                    "schema_context": "",
                })
            )
        try:
            ns = _exec_code(combine_code, extra_ns)
            r  = ns.get("result")
            if r is None:
                raise ValueError("'result' not created.")
            return r
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                return step_results[0]   # graceful degradation
