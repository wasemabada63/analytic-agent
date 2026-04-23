<div align="center">

<h1>🤖 DEEB Analytics Bot</h1>

<p><strong>Agentic RAG · SQL on Live DWH · Compound Queries · Arabic / English · Powered by Groq</strong></p>

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=flat-square)
![Groq LLM](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-6.x-purple?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-red?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-DWH-blue?style=flat-square&logo=postgresql)
![Plotly](https://img.shields.io/badge/Plotly-Charts-lightblue?style=flat-square)
![MIT License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

</div>

---

## 📋 Table of Contents

1. What Is This?
2. Quick Start
3. Project Structure
4. Architecture & Pipeline Flow
5. Features & Capabilities
6. Data Model (DWH)
7. API Response Schema
8. Configuration Reference
9. Export Capabilities
10. Known Limitations

---

## 🧠 What Is This?

**DEEB Analytics Bot** is a fully agentic, bilingual (Arabic + English) retail analytics assistant backed by a **live PostgreSQL Data Warehouse** (Supabase, schema `dwh1`). You type a natural-language question — in either language — and the system:

- Understands your intent and corrects typos
- Resolves pronouns and follow-up references from conversation history
- Decomposes compound questions into sub-queries
- Generates, validates, and executes **SQL** against the live DWH autonomously
- Delivers `chart_html` + `chart_json` for frontend rendering (no iframes, no local files)
- Delivers actionable business recommendations
- Caches semantically similar questions to avoid redundant LLM calls
- Remembers previous questions in the session

No dashboards. No manual SQL. Just questions.

---

## 🚀 Quick Start

### 1 — Clone & enter the repo

```bash
git clone https://github.com/Deeb-AI/AI-.git
cd deeb-analytics
```

### 2 — Create a virtual environment

```bash
# conda
conda create -n deeb python=3.11 -y
conda activate deeb

# or plain venv
python -m venv .venv && .venv\Scripts\activate   # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Set up environment variables

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

Open `.env` and fill in:

```env
GROQ_API_KEY=gsk_your_key_here

DWH_USER=your_supabase_user
DWH_PASS=your_supabase_password
DWH_HOST=your_supabase_host
DWH_PORT=6543
DWH_NAME=postgres
DWH_SCHEMA=dwh1
```

Get a free Groq key at → **[console.groq.com](https://console.groq.com)**

### 5 — Launch

```bash
python app.py
```

Open your browser at → **http://127.0.0.1:8080**

> The FAISS DWH schema index is already included (`data/faiss_dwh_index/`). No CSV files needed — all data is read live from the DWH.

---

## 📁 Project Structure

```
deeb-analytics/
│
├── app.py                    ← Gradio UI entry point  (python app.py)
├── test_chart.py             ← Smoke-test for chart_html / chart_json output
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/                      ← Core intelligence layer
│   ├── config.py             · DWH engine, FAISS index, date range constants
│   ├── llm.py                · ChatGroq singleton + token tracker
│   ├── prompts.py            · All LangChain prompt templates
│   ├── session.py            · Conversation memory, semantic cache, state
│   ├── helpers.py            · _strip_fences, _get_cache_key
│   ├── intent.py             · Spelling, rewrite, chitchat, classify, decompose, chart-edit
│   ├── executor.py           · SQL validator, SQL executor, compound combine step
│   ├── export.py             · PDF report, NL summary, recommendations, follow-ups
│   └── pipeline.py           · ask_retail_rag_ui() — the 8-step streaming pipeline
│
├── utils/                    ← Shared utilities
│   ├── arabic.py             · fix_arabic(), _ar_str(), reportlab font setup
│   ├── formatting.py         · KPI card, HTML table, number formatter, compound subplot chart
│   └── logger.py             · JSONL query logging
│
└── data/                     ← Index files only (no CSVs — data lives in DWH)
    └── faiss_dwh_index/
        ├── index.faiss
        └── index.pkl
```

---

## 🔄 Architecture & Pipeline Flow

Every question passes through **12 sequential layers** before a response is returned:

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1 · Spelling Correction                      │
│  Fixes Arabic/English typos before anything else    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2 · Reference Resolution                     │
│  Expands pronouns and short follow-ups using        │
│  conversation history stored in session.py          │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Chitchat?         Analytics?
              │                 │
    ┌─────────┘       ┌─────────┘
    ▼                 ▼
Friendly        ┌─────────────────────────────────────┐
reply &         │  LAYER 3 · Query Rewriter            │
return          │  Arabic question → precise English   │
                │  intent (preserves numbers, filters) │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 4 · Intent Classifier         │
                │  Detects: ranking | trend |          │
                │  distribution | comparison |         │
                │  correlation | detail                │
                │  → chart_type, top_n, time_filter,   │
                │    needs_chart, dimension            │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 5 · Semantic Cache Lookup     │
                │  MiniLM embedding + cosine sim       │
                │  4-level filter: lang → top_n →      │
                │  time_filter → dimension             │
                │  Hit → return instantly (0 tokens)   │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 6 · Schema Retrieval (FAISS)  │
                │  Top-k semantic search over the      │
                │  DWH schema index (dwh1 tables)      │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 7 · Query Decomposer          │
                │  Simple → direct SQL execution       │
                │  Compound → multi-step plan:         │
                │    Step 1 SQL → Step 2 SQL → Combine │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 8 · SQL Agent Loop            │
                │  Generate SQL → Validate (no DDL)   │
                │  → Execute on DWH → Sanity Check     │
                │  → auto-fix & retry (up to 3 tries) │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 9 · NL Summary (streamed)     │
                │  2-3 sentence plain-language         │
                │  interpretation of the result        │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 10 · Plotly Chart Agent       │
                │  Intent + data-driven chart type     │
                │  Compound → subplot (one panel/step) │
                │  Returns chart_html + chart_json     │
                │  (no files, no iframes)              │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 11 · Business Recommendations │
                │  Accumulates context across queries  │
                │  — avoids repeating prior points     │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  LAYER 12 · Follow-up Generator      │
                │  3 specific clickable follow-up      │
                │  questions based on actual result    │
                └──────────────┬──────────────────────┘
                               │
                               ▼
              Result Table + chart_json + Recommendations
                   + Follow-ups + Pipeline Log
```

---

## ✨ Features & Capabilities

### 🗄️ Live SQL on PostgreSQL DWH

The agent no longer reads CSV files. Every query runs as a validated `SELECT` against the live Supabase DWH (`dwh1` schema):

- LLM generates SQL based on FAISS-retrieved schema context
- SQL is sanitized before execution — no `INSERT`, `UPDATE`, `DELETE`, `DROP`, or any DDL/DML allowed
- Row cap enforced at 10,000 rows
- Auto-retry loop: if SQL fails, the error is fed back to the LLM for self-correction (up to 3 attempts)
- Connection pool with `pool_pre_ping` for reliability

---

### ⚡ Semantic Cache

Semantically similar questions return instantly from cache — zero LLM calls, zero tokens.

- Questions are embedded with `all-MiniLM-L6-v2` (multilingual MiniLM)
- **4-level exact-match filter** before cosine similarity check:
  1. `lang` — Arabic queries never match English cache entries
  2. `top_n` — "Top 5" never matches "Top 10"
  3. `time_filter` — "2024" never matches "2025"
  4. `dimension` — "products" never matches "categories"
- Cosine similarity threshold: **0.70**
- Cache stores up to 500 entries; auto-evicts oldest on overflow

---

### 🔀 Compound Query Decomposition

Multi-part questions are detected automatically and broken into SQL sub-steps.

| Example Question | What Happens |
|---|---|
| `"قارن إيرادات 2024 مقابل 2023 لكل فئة"` | Step 1: 2024 SQL → Step 2: 2023 SQL → Combine: pct_change |
| `"أكثر 5 منتجات ثم اتجاه مبيعاتهم شهرياً"` | Step 1: top-5 SQL → Step 2: monthly trend filtered by step-1 |
| `"أكثر 5 منتجات واكثر 5 فئات بالإيرادات"` | Step 1 SQL + Step 2 SQL → display_separately (stacked tables + subplot chart) |

Supported combination strategies: `merge_on_key`, `subtract`, `pct_change`, `display_separately`, `filter_by_step1`.

---

### 📊 Frontend-Ready Chart Output

Charts are never rendered server-side or saved to disk. Every response includes:

| Field | Content | Frontend usage |
|---|---|---|
| `chart_html` | Self-contained Plotly HTML snippet | `div.innerHTML = chart_html` |
| `chart_json` | Raw Plotly JSON spec (`data` + `layout`) | `Plotly.react(el, spec.data, spec.layout)` |

Both are `""` when no chart is applicable (e.g. single KPI result).

**Frontend rendering (chart_json):**

```javascript
const spec = JSON.parse(response.chart_json);
Plotly.react("chart-div", spec.data, spec.layout);
```

**Compound queries** produce a subplot figure — one panel per sub-result — in a single `chart_json`.

---

### 🎯 Intent-Guided + Data-Driven Visualization

The intent classifier runs **before** SQL generation, and the result data overrides chart type when needed:

- Ranking → horizontal bar chart
- Trend → line chart with chronological x-axis
- Distribution → pie (≤5 slices) or donut (>5)
- **Data override**: if result contains time columns (`month`, `year`, `date`, `week`) and intent was misclassified, pipeline forces LINE chart regardless
- All Arabic text in charts is reshaped and BiDi-corrected before rendering
- **Chart-edit intercept**: user can say "flip to bar" / "recolor" without re-running the SQL

---

### 🛡️ Multi-Layer SQL Safety

Generated SQL is never blindly executed:

1. **DDL/DML block** — regex blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `COPY`, `MERGE` and more
2. **Single statement only** — semicolons inside the query are rejected
3. **SELECT/WITH only** — any non-SELECT root is rejected
4. **Row cap** — `LIMIT 10000` enforced; user-supplied higher limits are capped
5. **Read-only transaction** — connection-level `SET TRANSACTION READ ONLY` as defense-in-depth
6. **Sanity check** — detects silent wrong-answer bugs:
   - All-zero numeric columns → forces retry
   - Ranking query returns 1 row → forces retry
   - Trend query has no time column → forces retry

---

### 💬 Bilingual with True Context Memory

- Accepts mixed Arabic/English questions in the same session
- Responds in whichever language the user wrote in
- Remembers the last 6 Q&A turns including result columns
- Resolves pronouns across turns (`"والان فقط 2024"` after `"اعرض اكثر 10 منتجات"` → correctly adds `year = 2024` filter to the SQL)

---

### 💡 Accumulating Business Recommendations

- Each new recommendation **builds on** previous ones rather than repeating them
- The LLM is shown the full history of prior recommendations and told to add new insights only
- Stored as dicts and rendered in full in the PDF report

---

### 📝 Streamed Natural Language Summary

After every successful SQL execution, a 2-3 sentence plain-language interpretation streams token-by-token to the UI — so users immediately understand the data without reading the table.

---

### 📄 Full PDF Report

One click generates a professional report containing:

- Complete query history with timestamps and result shapes
- All charts embedded as high-resolution PNGs (requires `kaleido`)
- Every business recommendation from the session, untruncated
- Arabic text rendered with proper glyph shaping via BiDi algorithm

---

### 📊 Token Tracking (No Double-Counting)

The `_TokenTracker` callback uses **OR logic** — reads from `generation_info.usage` first, falls back to `response_metadata.token_usage` only if empty. Prevents the double-counting bug that inflates totals when Groq populates both fields.

---

## 🗃️ Data Model (DWH)

All data lives in PostgreSQL schema `dwh1` (Supabase). Star schema:

```
dim_date ──────────────────────────────────────────────────────────┐
  date_key (PK)                                                     │
  full_date, year, month, month_name, week, day, quarter           │
                                                                    │ join on order_date_key
fact_order_item ────────────────────────────────────────────────────┤
  order_item_key (PK)                                               │
  order_id                                                          │
  order_date_key (FK → dim_date)                                    │
  product_key    (FK → dim_product)                                 │
  total_amount   ← USE THIS for revenue                             │
  quantity                                                          │
  unit_price                                                        │
                                                                    │
dim_product ───────────────────────────────────────────────────────┤
  product_key (PK)                                                  │
  name (Arabic), en_name (English)                                  │
  category_key   (FK → dim_category)                               │
  subcategory_key (FK → dim_subcategory)                           │
                                                                    │
dim_category ──────────────────────────────────────────────────────┤
  category_key (PK)                                                 │
  name (Arabic), en_name (English)                                  │
                                                                    │
dim_subcategory ────────────────────────────────────────────────────┘
  subcategory_key (PK)
  name (Arabic), en_name (English)
  category_key (FK → dim_category)
```

> ⚠️ **No CSV files are used.** `orders`, `products`, `order_items`, `categories`, `subcategories` variables are set to `None` in `config.py` — they are legacy stubs only.

---

## 📡 API Response Schema

Every pipeline call is a generator. The **final chunk** (`done=True`) contains:

| Field | Type | Description |
|---|---|---|
| `chat_text` | `str` | NL summary for the chat bubble |
| `result_html` | `str` | Styled HTML table — drop into a `<div>` |
| `chart_html` | `str` | Self-contained Plotly HTML snippet |
| `chart_json` | `str` | Plotly JSON spec — primary field for frontend rendering |
| `reco_text` | `str` | Business recommendations (markdown) |
| `followup` | `list[str]` | 3 suggested follow-up questions |
| `log` | `str` | Step-by-step pipeline log (debug) |
| `tokens_used` | `int` | LLM tokens consumed this call |
| `summary` | `str` | Alias for `chat_text` |
| `done` | `bool` | `True` on final chunk only — ignore intermediate chunks |

**Frontend needs:** `chat_text`, `result_html`, `chart_json`, `reco_text`, `followup`

---

## ⚙️ Configuration Reference

| Variable | Location | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` | Groq API key — required |
| `DWH_USER / DWH_PASS / DWH_HOST / DWH_PORT / DWH_NAME` | `.env` | PostgreSQL DWH credentials |
| `DWH_SCHEMA` | `.env` | Default `dwh1` |
| `DWH_STATEMENT_TIMEOUT_MS` | `.env` | Query timeout in ms (default: 30000) |
| `server_port` | `app.py` | Default `8080` |
| `MAX_HISTORY_TURNS` | `src/session.py` | Conversation turns kept in context (default: 6) |
| `_SEMANTIC_THRESHOLD` | `src/session.py` | Cosine similarity threshold for cache (default: 0.70) |
| `_SEMANTIC_MAX_ENTRIES` | `src/session.py` | Max semantic cache size (default: 500) |
| `max_retries` | `src/pipeline.py` | SQL + Plotly retry attempts (default: 3) |
| `top_k` | `src/pipeline.py` | FAISS schema chunks retrieved per query (default: 5) |
| `_MAX_ROWS` | `src/executor.py` | Maximum rows returned from DWH (default: 10,000) |

---

## 📤 Export Capabilities

| Format | How | Location |
|---|---|---|
| **CSV (.csv)** | "Export CSV" button → file download | `last_result.csv` |
| **Chart HTML** | Inline in API response (`chart_html` field) | — |
| **Chart JSON** | Inline in API response (`chart_json` field) | — |
| **Chart PNG** | Embedded in PDF (requires `kaleido`) | `temp_chart_*.png` (auto-cleaned) |
| **PDF Report** | "PDF Report" button | `deeb_report_TIMESTAMP.pdf` |
| **Query Log** | Automatic, every query | `query_log.jsonl` |
| **Session** | "Save Session" button | `session.json` |


<div align="center">
<sub>Built as a graduation capstone project · Retail AI · 2025–2026</sub>
</div>
