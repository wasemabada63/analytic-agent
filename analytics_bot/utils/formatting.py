"""
formatting.py
=============
HTML display helpers: KPI card, styled table, number column formatter.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

# ── Table CSS ──────────────────────────────────────────────────
_TABLE_CSS = """
<style>
.deeb-table {border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px}
.deeb-table thead tr{background:#1a1a2e;color:#e0e0e0}
.deeb-table tbody tr:nth-child(even){background:#f4f6fb}
.deeb-table tbody tr:hover{background:#dbeafe}
.deeb-table th,.deeb-table td{padding:8px 12px;border:1px solid #ddd;text-align:right;direction:rtl}
</style>
"""


# ── Number formatter ───────────────────────────────────────────
def _format_number_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy of df with monetary/large numbers formatted as K/M KWD."""
    disp = df.copy()
    for col in disp.select_dtypes(include="number").columns:
        col_l = col.lower()
        if any(kw in col_l for kw in ["revenue", "price", "total", "amount", "fee", "sales", "_kwd", "_jd", "value"]):
            def _fmt(val, _col=col):
                if pd.isna(val):
                    return val
                av = abs(val)
                if av >= 1_000_000:
                    return f"{val / 1_000_000:.2f}M KWD"
                if av >= 1_000:
                    return f"{val / 1_000:.1f}K KWD"
                return f"{round(val, 2)} KWD"
            disp[col] = disp[col].apply(_fmt)
    return disp


# ── KPI card (single-value results) ───────────────────────────
def _make_kpi_card(df: pd.DataFrame) -> str:
    """Render a single-value result as a big gradient KPI metric tile."""
    cards_html = ""
    for col in df.columns:
        val = df.iloc[0][col]
        numeric = isinstance(val, (int, float)) and not isinstance(val, bool)
        if numeric:
            av = abs(val)
            is_money = any(
                k in col.lower()
                for k in ["revenue", "price", "fee", "total", "sales", "_kwd", "_jd", "amount", "value"]
            )
            if av >= 1_000_000:
                disp = f"{val / 1_000_000:.2f}M KWD" if is_money else f"{val / 1_000_000:.2f}M"
            elif av >= 1_000:
                disp = f"{val / 1_000:.1f}K KWD" if is_money else f"{val / 1_000:.1f}K"
            else:
                disp = (
                    f"{round(val, 2)} KWD" if is_money
                    else (f"{val:,}" if isinstance(val, int) else f"{round(val, 2)}")
                )
        else:
            disp = str(val)

        cards_html += (
            f"<div style='background:linear-gradient(135deg,#1a1a2e 0%,#2563eb 100%);"
            f"border-radius:16px;padding:28px 36px;margin:10px auto;display:inline-block;"
            f"min-width:240px;text-align:center;box-shadow:0 4px 24px rgba(37,99,235,0.3)'>"
            f"<div style='color:#93c5fd;font-size:0.85rem;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:0.06em;margin-bottom:10px'>{col}</div>"
            f"<div style='color:#fff;font-size:2.8rem;font-weight:800;line-height:1.1'>{disp}</div>"
            f"</div>"
        )
    return f"<div style='display:flex; flex-wrap:wrap; justify-content:center; gap:16px; padding:24px'>{cards_html}</div>"


# ── Styled HTML table ──────────────────────────────────────────
def _df_to_html(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a styled HTML table with a Copy button."""
    uid = f"tbl_{abs(hash(str(list(df.columns))))}"
    table = df.head(200).to_html(index=False, classes="deeb-table", border=0)
    # Insert id on the <table> tag for clipboard selection
    table = table.replace("<table ", f'<table id="{uid}" ', 1)
    copy_btn = (
        f"<div style='margin-bottom:6px'>"
        f"<button onclick=\"(function(btn){{var t=document.getElementById('{uid}');"
        f"var r=document.createRange();r.selectNode(t);"
        f"window.getSelection().removeAllRanges();window.getSelection().addRange(r);"
        f"document.execCommand('copy');window.getSelection().removeAllRanges();"
        f"btn.textContent='✅ Copied!';setTimeout(()=>btn.textContent='📋 Copy table',1600);"
        f"}})(this)\" "
        f"style='padding:4px 14px;border-radius:6px;border:1px solid #bfdbfe;"
        f"background:#f0f7ff;cursor:pointer;font-size:0.82rem'>"
        f"📋 Copy table</button></div>"
    )
    return _TABLE_CSS + copy_btn + table


# ── Compound subplot chart ─────────────────────────────────────
_TIME_KWS = ["month", "year", "date", "week", "day", "quarter"]


def _build_compound_chart(step_results: list, step_labels: list) -> Optional[object]:
    """
    Build a Plotly subplot figure for display_separately compound queries.
    One panel per step result — line for time-series, hbar for categorical.
    Returns a Figure or None if nothing is chartable.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from utils.arabic import fix_arabic

        chartable = []
        for df, label in zip(step_results, step_labels):
            if df is None or df.empty or len(df) <= 1:
                continue
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                continue
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            time_cols = [c for c in df.columns if any(kw in c.lower() for kw in _TIME_KWS)]
            chartable.append({
                "df":       df,
                "label":    label[:55],
                "num_col":  num_cols[0],
                "cat_col":  cat_cols[0] if cat_cols else None,
                "time_col": time_cols[0] if time_cols else None,
                "is_time":  bool(time_cols),
            })

        if not chartable:
            return None

        n = len(chartable)
        fig = make_subplots(rows=1, cols=n, subplot_titles=[c["label"] for c in chartable])

        for i, c in enumerate(chartable):
            df, num_col, col = c["df"], c["num_col"], i + 1
            if c["is_time"] and c["time_col"]:
                x_vals = df[c["time_col"]].astype(str)
                fig.add_trace(
                    go.Scatter(x=x_vals, y=df[num_col], mode="lines+markers", showlegend=False),
                    row=1, col=col,
                )
            elif c["cat_col"]:
                labels = df[c["cat_col"]].astype(str).apply(fix_arabic)
                fig.add_trace(
                    go.Bar(x=df[num_col], y=labels, orientation="h", showlegend=False),
                    row=1, col=col,
                )

        fig.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=10))
        return fig
    except Exception:
        return None
