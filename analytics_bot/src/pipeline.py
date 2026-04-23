"""
pipeline.py
===========
The main 8-step agentic RAG orchestrator.

Steps:
  0. Intercept for spelling/references/chitchat.
  1. Rewrite for clear intent.
  2. Classify intent.
  3. Retrieve schema via FAISS.
  4. Execute SQL (handling simple vs compound queries).
  5. Format tables / generate text summaries.
  6. Generate charts via Plotly.
  7. Generate business recommendations.
  8. Suggest follow-up questions.
"""
from typing import Dict, Any, Optional
import os
import uuid

import pandas as pd
from langchain_core.output_parsers import StrOutputParser

import analytics_bot.src.session as _sess
from analytics_bot.src.config import BASE_DIR
from analytics_bot.src.llm import llm, _token_tracker
from analytics_bot.src.intent import (
    _correct_spelling,
    _resolve_references,
    _is_analytics_query,
    _get_chitchat_response,
    _classify_intent,
    _decompose_query,
    _rewrite_query,
    _is_chart_edit,
    _apply_chart_edit,
)
from analytics_bot.src.executor import (
    _build_schema_context,
    _exec_code,
    _exec_sql,
    _sanity_check_result,
    _run_sql_step,
    _combine_step_results,
)
from analytics_bot.src.prompts import (
    SQL_PROMPT, SQL_FIX_PROMPT,
    PLOTLY_PROMPT, PLOTLY_FIX_PROMPT,
)
from analytics_bot.src.export import (
    _generate_business_recommendation,
    _generate_followup_questions,
    _generate_summary_fallback,
)
from analytics_bot.utils.formatting import _df_to_html, _make_kpi_card, _format_number_cols, _build_compound_chart
from analytics_bot.utils.logger import _log_query
from analytics_bot.src.helpers import _strip_fences, _get_cache_key


def _log(msg: str) -> None:
    print(msg)


def ask_retail_rag_ui(
    question: str,
    use_viz: bool = True,
    use_reco: bool = True,
    use_cache: bool = True,
    max_retries: int = 3,
):
    """
    Main entry point for the Gradio interface.
    Yields intermediate dicts to support streaming of the NL summary.
    """
    tokens_before = _token_tracker.total
    log_lines = []

    def _log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    _log(f"\n{'='*60}\n👤 USER: {question}\n{'='*60}")

    # ── Chart-edit intercept ─────────────────────────────────────
    if _sess._last_plotly_code and _is_chart_edit(question):
        _log("🎨 Interpreted as a chart-edit request.")
        try:
            new_code = _apply_chart_edit(
                question, _sess._last_plotly_code, _sess._last_result
            )
            ns = _exec_code(new_code, {"result": _sess._last_result})
            fig = ns.get("fig")
            if not fig:
                raise ValueError("No 'fig' produced.")
            _sess._last_fig = fig
            _sess._last_plotly_code = new_code
            chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            chart_json = fig.to_json()
            yield dict(
                chat_text="✅ Chart updated.",
                result_html=_make_kpi_card(_sess._last_result) if _sess._last_result.shape[0] == 1 else _df_to_html(_format_number_cols(_sess._last_result)),
                chart_html=chart_html, chart_json=chart_json, reco_text="", followup=[],
                log="\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before,
                summary="Chart updated.", done=True
            )
            return
        except Exception as e:
            _log(f"   ⚠️ Chart edit failed: {e}. Falling back to normal pipeline.")

    # ── Step 0: Spelling & Intercepts ────────────────────────────
    corrected = _correct_spelling(question)
    if corrected != question:
        _log(f"🔤 Corrected: {corrected}")
    
    rewritten = _resolve_references(corrected)
    if rewritten != corrected:
        _log(f"🔗 Resolved: {rewritten}")

    if not _is_analytics_query(rewritten):
        _log("💬 Intercepted as chitchat.")
        ans = _get_chitchat_response(rewritten)
        yield dict(
            chat_text=ans, result_html="", chart_html="",
            reco_text="", followup=[], log="\n".join(log_lines),
            tokens_used=_token_tracker.total - tokens_before,
            summary=ans, done=True
        )
        return

    # ── Cache check ──────────────────────────────────────────────
    cache_key = _get_cache_key(rewritten, use_viz, use_reco)
    if use_cache and cache_key in _sess._query_cache:
        _log("⚡ Cache hit!")
        c = _sess._query_cache[cache_key]
        _sess._add_to_history(question, rewritten, list(c["result"].columns) if c["result"] is not None else [], success=True)
        _sess._add_query_to_history(question, "success (cache)", f"{len(c['result'])}x{len(c['result'].columns)}" if c["result"] is not None else "N/A")
        
        # Determine HTML representation
        res = c["result"]
        res_html = _make_kpi_card(res) if res.shape[0] == 1 else _df_to_html(_format_number_cols(res))
        
        yield dict(
            chat_text   = c["summary"],
            result_html = res_html,
            chart_html  = c["chart_html"],
            chart_json  = c.get("chart_json", ""),
            reco_text   = c["reco"],
            followup    = c["followup"],
            log         = "\n".join(log_lines),
            tokens_used = _token_tracker.total - tokens_before,
            summary     = c["summary"],
            done        = True
        )
        return

    # ── Step 1 & 2: Rewrite & Intent ─────────────────────────────
    _question_lang = _sess._detect_lang(question)
    yield dict(chat_text="⏳ Analyzing query...", result_html="", chart_html="", reco_text="", followup=[], log="\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary="", done=False)
    rewritten_final = _rewrite_query(rewritten)
    if rewritten_final != rewritten:
        _log(f"📝 Final rewrite: {rewritten_final}")
        rewritten = rewritten_final

    intent = _classify_intent(rewritten)
    _log(f"🎯 Intent: {intent['intent_type']} | top_n={intent['top_n']} | time={intent['time_filter']}")

    history_context = "\n".join(
        f"User: {h.get('question')}\nBot generated columns: {h.get('columns', [])}"
        for h in _sess._conversation_history[-3:]
    )
    intent_hint = (
        f"Intent: {intent['intent_type']}, time_filter: {intent['time_filter']}, "
        f"top_n: {intent['top_n']}. Follow these constraints."
    )
    chart_hint = (
        f"This should be a **{(intent.get('chart_type') or 'vbar').upper()}** chart "
        f"(intent: {intent['intent_type']}). Follow this unless data clearly contradicts it."
    )

    # ── Semantic cache lookup (after intent — uses top_n + time_filter for exact match) ──
    if use_cache:
        c = _sess._semantic_lookup(
            rewritten, use_viz, use_reco,
            lang=_question_lang,
            top_n=intent.get("top_n"),
            time_filter=intent.get("time_filter"),
            dimension=intent.get("dimension", "general"),
        )
        if c is not None:
            _log("🧠 Semantic cache hit!")
            _sess._add_to_history(question, rewritten, list(c["result"].columns) if c["result"] is not None else [], success=True)
            _sess._add_query_to_history(question, "success (semantic cache)", f"{len(c['result'])}x{len(c['result'].columns)}" if c["result"] is not None else "N/A")
            res = c["result"]
            res_html = _make_kpi_card(res) if res.shape[0] == 1 else _df_to_html(_format_number_cols(res))
            yield dict(
                chat_text   = c["summary"],
                result_html = res_html,
                chart_html  = c["chart_html"],
                chart_json  = c.get("chart_json", ""),
                reco_text   = c["reco"],
                followup    = c["followup"],
                log         = "\n".join(log_lines),
                tokens_used = _token_tracker.total - tokens_before,
                summary     = c["summary"],
                done        = True
            )
            return

    # ── Step 3: Schema retrieval ─────────────────────────────────
    yield dict(chat_text="⏳ Retrieving schema context...", result_html="", chart_html="", reco_text="", followup=[], log="\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary="", done=False)
    schema_context = _build_schema_context(rewritten, top_k=6)

    # ── Step 3.5: Decompose ──────────────────────────────────────
    _log("🔍 Analyzing query complexity…")
    plan = _decompose_query(rewritten)

    result: Optional[pd.DataFrame] = None

    if plan["is_compound"]:
        _log(f"   🔀 Compound — {len(plan['steps'])} sub-steps [{plan['combination']}]")
        step_results = []
        attempts_total = 0

        for i, step in enumerate(plan["steps"]):
            _log(f"\n━━ Sub-step {i+1}: {step}")
            step_df, step_sql = _run_sql_step(
                step, schema_context, intent_hint,
                history_context, max_retries, i + 1,
            )
            if step_sql:
                _log(f"   SQL: {step_sql[:200]}{'...' if len(step_sql) > 200 else ''}")
            attempts_total += 1

            if step_df is None:
                _log(f"❌ Sub-step {i+1} failed.")
                _log_query(question, "failed", f"Sub-step {i+1} failed", attempts_total)
                _sess._add_to_history(question, rewritten, [], success=False)
                _sess._add_query_to_history(question, "failed", "0x0")
                yield dict(
                    chat_text="❌ Sub-step failed. Please rephrase.",
                    result_html="", chart_html="", reco_text="", followup=[],
                    log="\n".join(log_lines),
                    tokens_used=_token_tracker.total - tokens_before,
                    summary="", done=True
                )
                return

            step_results.append(step_df)

        result = _combine_step_results(question, plan["steps"], plan["combination"], step_results, max_retries)

    else:
        _log("   ➡️  Simple query.")

        # ── Step 4: SQL loop ─────────────────────────────────────
        yield dict(chat_text="⏳ Generating SQL query...", result_html="", chart_html="", reco_text="", followup=[], log="\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary="", done=False)
        sql: Optional[str] = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                sql = _strip_fences(
                    (SQL_PROMPT | llm | StrOutputParser()).invoke({
                        "schema_context":  schema_context,
                        "question":        question,
                        "intent_hint":     intent_hint,
                        "history_context": history_context,
                    })
                )
            else:
                _log(f"🔁 Retry SQL attempt {attempt}…")
                sql = _strip_fences(
                    (SQL_FIX_PROMPT | llm | StrOutputParser()).invoke({
                        "sql":            sql,
                        "error":          str(last_error),
                        "question":       question,
                        "schema_context": schema_context,
                    })
                )
            try:
                yield dict(chat_text="⏳ Fetching data from database...", result_html="", chart_html="", reco_text="", followup=[], log="\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary="", done=False)
                result = _exec_sql(sql)
                is_valid, warning = _sanity_check_result(result, intent)
                if not is_valid:
                    raise ValueError(warning)
                _log(f"   SQL: {sql[:300]}{'...' if len(sql) > 300 else ''}")
                _log(f"   ✅ SQL success (attempt {attempt}), shape: {result.shape}")
                break
            except Exception as e:
                last_error = e
                _log(f"⚠️  Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    _log("❌ Max retries reached — generating text summary…")
                    summary = _generate_summary_fallback(
                        question, str(last_error), schema_context,
                    )
                    _log_query(question, "failed", str(last_error), max_retries)
                    _sess._add_to_history(question, rewritten, [], success=False)
                    _sess._add_query_to_history(question, "failed", "0x0")
                    yield dict(
                        chat_text=(
                            f"⚠️ Could not generate exact data. "
                            f"Here is a qualitative answer:\n\n{summary}"
                        ),
                        result_html="", chart_html="", reco_text="", followup=[],
                        log="\n".join(log_lines),
                        tokens_used=_token_tracker.total - tokens_before,
                        summary="", done=True
                    )
                    return

    if result is None:
        _sess._add_to_history(question, rewritten, [], success=False)
        _sess._add_query_to_history(question, "failed", "0x0")
        yield dict(
            chat_text="❌ Query failed entirely.",
            result_html="", chart_html="", reco_text="", followup=[],
            log="\n".join(log_lines),
            tokens_used=_token_tracker.total - tokens_before,
            summary="", done=True
        )
        return

    # ── display_separately: normalize list → stacked tables + combined preview ──
    _summary_preview_override = None
    _step_results_for_chart: list = []
    result_html = ""
    if isinstance(result, list):
        result_list = result
        _step_results_for_chart = result_list
        _time_kws = ["month", "year", "date", "week", "day", "quarter"]
        primary_result = next(
            (df for df in result_list
             if any(kw in c.lower() for c in df.columns for kw in _time_kws)),
            result_list[0],
        )
        html_parts = []
        for step, df in zip(plan["steps"], result_list):
            label = (
                f"<div style='font-weight:700;padding:10px 0 4px;"
                f"color:#1a1a2e;font-size:0.95rem'>📊 {step[:70]}</div>"
            )
            html_parts.append(label + _df_to_html(_format_number_cols(df)))
        result_html = (
            "<div style='margin-bottom:20px'>"
            + "</div><div style='margin-bottom:20px'>".join(html_parts)
            + "</div>"
        )
        preview_parts = []
        for i, (step, df) in enumerate(zip(plan["steps"], result_list)):
            preview_parts.append(
                f"--- Result {i+1}: {step[:60]} ---\n"
                + df.head(10).to_string(index=False)
                + (f"\n... ({len(df)-10} more rows)" if len(df) > 10 else "")
            )
        _summary_preview_override = "\n\n".join(preview_parts)
        result = primary_result  # normalize for chart / cache / recos

    _sess._add_to_history(question, rewritten, list(result.columns) if result is not None else [], success=True)
    _sess._last_result = result

    # ── Step 5: Formatting ───────────────────────────────────────
    if not result_html:
        if result.shape[0] == 1:
            result_html = _make_kpi_card(result)
        else:
            disp_df = _format_number_cols(result)
            result_html = _df_to_html(disp_df)

    # ── Step 5a: NL Summary ────────────────────────────────────────
    _log("📝 Generating natural language summary…")
    summary_chunks = []
    try:
        from analytics_bot.src.export import _generate_nl_summary_stream
        for chunk in _generate_nl_summary_stream(question, result, preview_override=_summary_preview_override):
            summary_chunks.append(chunk)
            current_summary = "".join(summary_chunks)
            yield dict(
                chat_text   = current_summary + " ▌",
                result_html = result_html,
                chart_html  = "",
                reco_text   = "",
                followup    = [],
                log         = "\n".join(log_lines),
                tokens_used = _token_tracker.total - tokens_before,
                summary     = current_summary,
            )
        summary = "".join(summary_chunks)
    except Exception as e:
        summary = f"⚠️ Summary unavailable: {e}"
    _log("   ✅ Summary ready.")

    yield dict(chat_text=summary + "\\n\\n⏳ Generating chart...", result_html=result_html, chart_html="", reco_text="", followup=[], log="\\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary=summary, done=False)
    # ── Step 6: Visualization ────────────────────────────────────
    chart_html = ""
    chart_json = ""
    plotly_code = ""
    png_path = None
    _has_time_col = any(
        kw in c.lower()
        for c in result.columns
        for kw in ["month", "year", "date", "week", "day", "quarter"]
    )
    _should_chart = intent["needs_chart"] or (_has_time_col and len(result) > 2)
    if _has_time_col and len(result) > 2 and intent["intent_type"] not in ("trend", "comparison"):
        chart_hint = (
            "This should be a **LINE** chart — the data has a time dimension "
            "(month/year/date columns). Put time on the x-axis and the metric on the y-axis."
        )
    if use_viz and _step_results_for_chart:
        _log("📊 Generating compound subplot chart…")
        fig = _build_compound_chart(_step_results_for_chart, plan["steps"])
        if fig is not None:
            try:
                _sess._last_fig = fig
                _sess._last_plotly_code = ""
                chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                chart_json = fig.to_json()
                _log("   ✅ Compound chart generated.")
            except Exception as e:
                _log(f"   ⚠️ Compound chart failed: {e}")
        else:
            _log("   ℹ️ No chartable data in compound results.")

    elif use_viz and _should_chart and len(result) > 1:
        _log("📊 Generating Plotly chart…")
        preview = result.head(15).to_string(index=False)
        code = ""
        last_err = None
        for attempt in range(1, 3):
            try:
                if attempt == 1:
                    code = _strip_fences(
                        (PLOTLY_PROMPT | llm | StrOutputParser()).invoke({
                            "question":     question,
                            "data_preview": preview,
                            "columns":      list(result.columns),
                            "chart_hint":   chart_hint,
                        })
                    )
                else:
                    _log(f"🔁 Retry chart (attempt {attempt})…")
                    code = _strip_fences(
                        (PLOTLY_FIX_PROMPT | llm | StrOutputParser()).invoke({
                            "code":         code,
                            "error":        str(last_err),
                            "question":     question,
                            "data_preview": preview,
                            "columns":      list(result.columns),
                        })
                    )
                ns = _exec_code(code, {"result": result})
                fig = ns.get("fig")
                if not fig:
                    raise ValueError("No 'fig' variable found in code.")
                _sess._last_fig = fig
                _sess._last_plotly_code = code
                chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                chart_json = fig.to_json()
                # Save PNG for PDF export
                try:
                    import uuid
                    png_path = os.path.join(BASE_DIR, f"temp_chart_{uuid.uuid4().hex[:8]}.png")
                    fig.write_image(png_path, width=900, height=460)
                    _sess._session_charts.append(png_path)
                except Exception as e:
                    _log(f"   ⚠️ Could not save PNG for PDF: {e}")
                _log("   ✅ Chart generated.")
                break
            except Exception as e:
                last_err = e
                _log(f"   ⚠️ Chart failed: {e}")

    yield dict(chat_text=summary + "\\n\\n⏳ Generating business recommendations...", result_html=result_html, chart_html=chart_html, reco_text="", followup=[], log="\\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary=summary, done=False)
    # ── Step 7: Business Recos ───────────────────────────────────
    reco_text = ""
    if use_reco:
        _log("💡 Generating business recommendations…")
        try:
            reco_text = _generate_business_recommendation(
                question, result, intent["intent_type"],
            )
            _log("   ✅ Recos ready.")
        except Exception as e:
            _log(f"   ⚠️ Recos failed: {e}")

    yield dict(chat_text=summary + "\\n\\n⏳ Generating follow-up questions...", result_html=result_html, chart_html=chart_html, reco_text=reco_text, followup=[], log="\\n".join(log_lines), tokens_used=_token_tracker.total - tokens_before, summary=summary, done=False)
    # ── Step 8: Follow-up Questions ──────────────────────────────
    followup = []
    _log("🔮 Generating follow-up questions…")
    try:
        followup = _generate_followup_questions(question, result)
        _log(f"   ✅ {len(followup)} follow-ups ready.")
    except Exception as e:
        _log(f"   ⚠️ Follow-up failed: {e}")

    # ── Final cleanup ────────────────────────────────────────────
    _log_query(question, "success")
    _sess._add_query_to_history(question, "success", f"{len(result)}x{len(result.columns)}")
    tokens_this_call = _token_tracker.total - tokens_before
    _log(f"   💸 Tokens used: {tokens_this_call:,}")

    full_reco = f"**Summary:**\n{summary}\n\n"
    if reco_text:
        full_reco += f"**Recommendations:**\n{reco_text}"
    _sess._add_recommendation_to_memory(question, full_reco.strip(), png_path)

    if use_cache:
        _sess._query_cache[cache_key] = {
            "result":     result,
            "chart_html": chart_html,
            "chart_json": chart_json,
            "reco":       reco_text,
            "followup":   followup,
            "summary":    summary,
        }
        _sess._semantic_store(
            rewritten, cache_key, use_viz, use_reco,
            lang=_question_lang,
            top_n=intent.get("top_n"),
            time_filter=intent.get("time_filter"),
            dimension=intent.get("dimension", "general"),
        )

    chat_text = summary.strip() if summary else (
        f"Result: {len(result)} rows × {len(result.columns)} columns."
    )

    yield dict(
        chat_text   = chat_text,
        result_html = result_html,
        chart_html  = chart_html,
        chart_json  = chart_json,
        reco_text   = reco_text,
        followup    = followup,
        log         = "\n".join(log_lines),
        tokens_used = tokens_this_call,
        summary     = summary,
        done        = True
    )
    return