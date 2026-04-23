"""
prompts.py
==========
All LangChain PromptTemplate definitions used in the pipeline.
No side-effects — pure data.
"""
from langchain_core.prompts import PromptTemplate

# ── Chitchat gate ─────────────────────────────────────────────
CHITCHAT_GATE_PROMPT = PromptTemplate(
    input_variables=["question", "history"],
    template="""
You are a classifier for a retail analytics assistant.
Decide whether the user input is:
  - "analytics" : any question about data, sales, orders, products, revenue, trends, categories, customers, reports, charts, statistics, or any request to show/display/analyze retail data.
               Also classify as "analytics" if it is a SHORT FOLLOW-UP or FILTER that clearly refines a previous analytics query (e.g. "for 2024 only", "خلال 2024 فقط", "والان لـ 2024", "top 5 only", "now filter by category").
  - "chitchat"  : greetings, general knowledge, history, geography, personal opinions, jokes, anything unrelated to the retail database.

Recent conversation history (use this to understand follow-up fragments):
{history}

User input: {question}

Respond with ONLY one word — either:  analytics   OR   chitchat
""")

# ── Chitchat response ─────────────────────────────────────────
CHITCHAT_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a friendly retail analytics assistant. The user has sent a non-analytics message.
Respond naturally and helpfully in the same language the user used.
If relevant, gently remind them that you can also answer retail data questions like sales trends, top products, revenue by category, etc.

User: {question}
Assistant:""")

# ── Query decomposition ───────────────────────────────────────
DECOMPOSE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a retail data analyst. Decide if the following question is SIMPLE or COMPOUND.

SIMPLE  = a single aggregation, ranking, or lookup that can be answered in one pandas operation.
COMPOUND = requires 2+ separate data operations, such as:
  - Comparing two time periods (2024 vs 2025)
  - Finding top-N AND showing their trend over time
  - Growth rate / percentage change between periods
  - "Items that declined / grew" (requires comparison)
  - Any question with AND / ثم / مقارنة / مقابل / نسبة النمو

Respond ONLY with valid JSON — no markdown, no explanation.

Question: {question}

Return exactly this structure:
{{
  "is_compound": <true or false>,
  "steps": [
    "<step 1: precise English description of first sub-query>",
    "<step 2: precise English description of second sub-query>"
  ],
  "combination": "<merge_on_key | subtract | pct_change | display_separately | filter_by_step1>"
}}
""")

# ── Combine sub-step results ──────────────────────────────────
COMBINE_PROMPT = PromptTemplate(
    input_variables=["question", "steps", "combination", "step_results_info"],
    template="""
You are a senior data analyst combining multiple intermediate DataFrames into a final answer.

Original question: {question}
Sub-steps performed: {steps}
Combination method: {combination}
Available DataFrames and their columns:
{step_results_info}

Write ONLY raw executable Python/pandas code that:
1. Combines step_result_0, step_result_1, ... using the combination method.
2. Stores the final combined answer as a DataFrame called `result`.
3. Prints `result`.

- merge_on_key   → inner merge on shared key, keep all value columns
- subtract       → if inputs are scalars, construct a 2-row DataFrame with a 'Period' column and a 'Value' column so it can be plotted as a bar chart. If they are tables, merge on key and compute (value_step0 - value_step1)
- pct_change     → if inputs are scalars, construct a 2-row DataFrame with a 'Period' column and a 'Value' column, plus a 'pct_change' column. If tables, merge on key and compute ((value_step1 - value_step0) / value_step0 * 100).round(2), rename to 'growth_pct'
- display_separately → result = pd.concat([step_result_0, step_result_1], axis=0, ignore_index=True) if they share columns, else return step_result_0
- filter_by_step1 → use step_result_0 values as a filter on step_result_1

CRITICAL: NEVER return a DataFrame with just a single scalar difference (e.g. [-150000]). ALWAYS return the original values from BOTH steps as separate rows (e.g. row 1 = 2024, row 2 = 2025) so the result can be visually plotted!

No markdown, no fences, no comments.
""")

# ── Intent classification ─────────────────────────────────────
INTENT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a retail data analyst. Classify the following user question and respond ONLY with valid JSON — no explanation, no markdown.

Question: {question}

Return exactly this structure:
{{
  "intent_type": "<ranking | trend | distribution | comparison | correlation | detail>",
  "chart_type":  "<hbar | line | vbar | pie | donut | histogram | scatter | area | table>",
  "needs_chart": <true or false>,
  "top_n":       <integer or null>,
  "time_filter": "<e.g. 2024, Q1 2024, or null>",
  "dimension":   "<the primary business entity being analyzed: product | category | subcategory | seller | customer | city | brand | total>"
}}
""")

# ── Query rewriter ────────────────────────────────────────────
REWRITE_PROMPT = PromptTemplate(
    input_variables=["question", "history"],
    template="""
You are a retail data analyst assistant. Rewrite the following user question into a clear, precise analytical intent in English.
Preserve all specific numbers, time filters, ranking limits, and entity names exactly as mentioned.
If the question is a SHORT FOLLOW-UP or FILTER (e.g. "for 2024 only", "خلال 2024 فقط", "والان لـ 2024"), use the conversation history to reconstruct the FULL analytical intent.
Do NOT answer the question — just rewrite it as a clear one-sentence English data analysis request.

Recent conversation history:
{history}

User question: {question}

Rewritten intent (one sentence):
""")

# ══════════════════════════════════════════════════════════════
# SQL PIPELINE (dwh1 star schema — new primary path)
# ══════════════════════════════════════════════════════════════

# ── SQL code generator ────────────────────────────────────────
SQL_PROMPT = PromptTemplate(
    input_variables=["schema_context", "question", "intent_hint", "history_context"],
    template="""
You are a senior data analyst writing PostgreSQL against a retail star-schema data warehouse.
Use ONLY the tables in the `dwh1` schema shown below.

SCHEMA CONTEXT (retrieved for this question):
{schema_context}

════════════════════════════════════════════════════
STAR SCHEMA CHEAT SHEET — dwh1
════════════════════════════════════════════════════

FACTS
  dwh1.fact_order_item                       -- main fact (233k rows); measures already pre-computed
    keys:    order_item_key (PK), customer_key, product_key, seller_key, category_key,
             brand_key, order_date_key, delivery_date_key, data_owner_key, order_id
    measures: unit_price, quantity, discount_amount, tax_amount, total_amount
    attrs:    order_status

DIMENSIONS
  dwh1.dim_product     product_key, product_id, name (ar), en_name, price, tax_rate, sku, currency
  dwh1.dim_category    category_key, category_id, category_name (ar), sub_category_id, sub_category_name
                       -- BOTH levels live in ONE row. There is NO separate sub_categories table.
  dwh1.dim_brand       brand_key, brand_id, brand_name (ar), brand_en_name
  dwh1.dim_customer    customer_key, customer_id, name, email, phone, city   (PII currently all NULL)
  dwh1.dim_seller      seller_key,   seller_id,   seller_name, email, phone, city (PII currently all NULL)
  dwh1.dim_date        date_key (int yyyymmdd), full_date, day, day_name, month, month_name, year
  dwh1.dim_data_owner  data_owner_key, data_owner_id, data_owner_name (multi-tenant; usually 1 tenant)

⚠️ CRITICAL RULES
1. REVENUE: use SUM(f.total_amount) from fact_order_item. The measure is already
   unit_price*quantity net of discount + tax. Never compute SUM(unit_price*quantity) manually.
2. JOINS: always join facts to dims on the *_key surrogate columns:
      f.product_key   = p.product_key
      f.category_key  = c.category_key
      f.brand_key     = b.brand_key
      f.customer_key  = cu.customer_key
      f.seller_key    = s.seller_key
      f.order_date_key = d.date_key     (use d.year, d.month, d.month_name, d.full_date)
3. CATEGORIES: one join to dim_category gives you BOTH category_name and sub_category_name.
   Do NOT try to join a sub_categories table.
4. DATES: filter on dim_date columns after joining (WHERE d.year = 2024, d.month = 6 etc).
   Do NOT parse date_key as a string.
5. DELIVERY DATES: delivery_date_key = 10000000 is a sentinel for 'unknown'.
   Exclude it: WHERE f.delivery_date_key <> 10000000 before joining delivery dates.
6. ORDER STATUS lifecycle (pick based on the question):
      waiting              -- pending (bulk of rows: ~124k)
      invoiced             -- finalized (~101k)
      preparing / storekeeper_received / storekeeper_finished / delivered / done
   For "completed / actual sales" default to: order_status IN ('done','delivered','invoiced').
   If the question does not specify, DO NOT filter on order_status unless the user asked for
   completed/pending only.
7. CURRENCY: all monetary amounts are in Kuwaiti Dinar (KWD / دينار كويتي).
   - Alias revenue columns with _kwd suffix (e.g. revenue_kwd, spend_kwd).
   - Round to 2 decimal places: ROUND(SUM(f.total_amount)::numeric, 2) AS revenue_kwd.
   - NEVER use $, USD, JD, SAR.
8. LIMIT: always add LIMIT to prevent huge result sets. Default LIMIT 100. For ranking with
   an intent_hint top_n use LIMIT = top_n. Never exceed LIMIT 10000.
9. SAFETY: produce exactly ONE read-only SELECT statement. No INSERT/UPDATE/DELETE/DDL,
   no multiple statements, no semicolons inside string literals.
10. Qualify every column with its table alias (f., p., c., d., b., cu., s.).
    Use short aliases: fact_order_item=f, dim_product=p, dim_category=c, dim_brand=b,
    dim_customer=cu, dim_seller=s, dim_date=d, dim_data_owner=o.
11. GROUP BY: include every non-aggregated selected column.
12. For trend intent, ORDER BY d.year, d.month (or d.full_date) ascending.
    For ranking intent, ORDER BY the aggregated measure DESC.

QUERY INTENT HINT:
{intent_hint}

CONVERSATION HISTORY:
{history_context}
════════════════════════════════════════════════════

User question: {question}

Write ONLY raw executable PostgreSQL. No markdown fences, no comments, no prose.
Return a single SELECT statement terminated by one semicolon.
""")

# ── SQL code fixer ────────────────────────────────────────────
SQL_FIX_PROMPT = PromptTemplate(
    input_variables=["sql", "error", "question", "schema_context"],
    template="""
You are a senior data analyst debugging PostgreSQL for a retail star-schema DWH (dwh1).
The following SQL was generated to answer: {question}

Relevant schema:
{schema_context}

CRITICAL REMINDERS:
- Revenue measure is already pre-computed: SUM(f.total_amount) from dwh1.fact_order_item.
- Join facts → dims on the surrogate *_key columns only.
- dim_category contains BOTH category and sub_category in one row. No sub_categories table.
- dim_date: date_key is YYYYMMDD integer. Filter on d.year/d.month after the join.
- delivery_date_key = 10000000 is a sentinel — exclude before joining delivery dates.
- Currency: KWD (Kuwaiti Dinar). Alias revenue columns with _kwd.
- Always qualify columns with the table alias (f., p., c., d., b., cu., s., o.).
- Exactly ONE read-only SELECT, with a LIMIT.

SQL that failed:
{sql}

Error: {error}

Return ONLY the corrected SQL — one SELECT, terminated with a semicolon. No markdown, no comments.
""")

# ── Plotly chart generator ────────────────────────────────────
PLOTLY_PROMPT = PromptTemplate(
    input_variables=["question", "data_preview", "columns", "chart_hint"],
    template="""
You are a senior data visualization expert specializing in Arabic retail analytics.

User question: {question}
Result columns: {columns}
Data preview:
{data_preview}

CHART HINT: {chart_hint}

Chart rules:
- LINE  (px.line): trend over time — sort by time column first.
- HBAR  (px.bar, orientation='h'): ranking / top-N — sort descending. Use x=numeric_col, y=name_col.
- VBAR  (px.bar): discrete group comparison with ≤ 6 categories.
- PIE   (px.pie): proportions ≤ 5 slices.
- DONUT (px.pie, hole=0.45): proportions > 5 slices.
- HISTOGRAM (px.histogram): continuous numeric distribution.
- SCATTER (px.scatter): correlation.
- ANIMATED TREND (px.bar or px.line with animation_frame): ONLY for 'trend' intent when data has
  a month or year column with ≥ 6 distinct values. Use animation_frame='month' or 'year'.

⚠️ STRICT CHART HINT ENFORCEMENT:
- If CHART HINT says HBAR → you MUST write: px.bar(df, x='<numeric_col>', y='<name_col>', orientation='h')
  Sort the dataframe by the numeric column descending BEFORE plotting. Never use vbar for ranking.
- If chart has > 6 categories on x-axis → always use HBAR (horizontal), never VBAR.
- Each row must map to exactly ONE bar. Never stack or group unless explicitly asked.

MANDATORY:
- Apply fix_arabic() on ALL Arabic text: titles, labels, tick labels, axis titles.
- fig.update_layout(font=dict(family="Arial"), title_x=0.5)
- Store figure as `fig`. End with fig.show().
- CURRENCY: label all monetary axes/titles with "دينار كويتي (KWD)" — NEVER use $, USD, JD, or SAR.

fix_arabic() and `result` DataFrame are already in memory.
Write ONLY raw Python/plotly code. No markdown, no fences.
""")

# ── Plotly chart fixer ────────────────────────────────────────
PLOTLY_FIX_PROMPT = PromptTemplate(
    input_variables=["code", "error", "question", "data_preview", "columns"],
    template="""
Fix this plotly code for: {question}
Columns: {columns}
Data preview: {data_preview}
Error: {error}

Code: {code}

fix_arabic() and `result` DataFrame are available. Write ONLY raw Python/plotly code.
""")

# ── Business recommendations ──────────────────────────────────
BUSINESS_RECO_PROMPT = PromptTemplate(
    input_variables=["question", "data_preview", "columns", "intent_type", "accumulated_recommendations"],
    template="""
You are a senior retail business analyst providing actionable recommendations to management.

Question answered: {question}
Intent: {intent_type}
Columns: {columns}
Data:
{data_preview}

ACCUMULATED RECOMMENDATIONS THIS SESSION:
{accumulated_recommendations}

Provide 3-5 concise, specific, actionable business recommendations.
- Reference specific numbers/names from the data.
- Build on accumulated recommendations — do NOT repeat same points.
- Be practical: stock, promotions, pricing, investigate, discontinue.
- CRITICAL LANGUAGE RULE: Look at the question text above. If it contains Arabic characters → write ALL recommendations in Arabic. If it is English → write entirely in English. NEVER mix languages.
- Format as a numbered list.
- CURRENCY: always state monetary figures in Kuwaiti Dinar (KWD / دينار كويتي). Never use $, USD, JD, or SAR.
""")

# ── Follow-up question generator ──────────────────────────────
FOLLOWUP_PROMPT = PromptTemplate(
    input_variables=["question", "result_preview", "columns"],
    template="""
You are a retail analytics assistant. The user just asked: "{question}"

The result had these columns: {columns}
Result preview:
{result_preview}

Suggest exactly 3 follow-up questions that a retail manager would naturally ask next.
CRITICAL LANGUAGE RULE: If the original question contains Arabic characters → write ALL 3 questions in Arabic. If it is English → write all 3 in English. NEVER mix languages.
Each question must be specific (reference actual values or entities from the result).

Respond ONLY with a JSON array of 3 strings. No markdown, no explanation. Example:
["question 1", "question 2", "question 3"]
""")

# ── Reference resolver ────────────────────────────────────────
REFERENCE_RESOLVE_PROMPT = PromptTemplate(
    input_variables=["question", "history"],
    template="""
You are a retail analytics assistant. The user sent a follow-up query that may use pronouns,
implicit references, or be a SHORT FILTER/MODIFIER of a previous query.

Conversation history:
{history}

Follow-up question: {question}

Rewrite as a FULLY SELF-CONTAINED question if any of these apply:
- Contains vague references: "نفس", "same", "them", "it", "ذات", "هم", "هو", "هي", "ذلك", "تلك"
- Is a SHORT FRAGMENT that adds a filter/modifier to the previous query (e.g. "والان خلال 2024 فقط" → expand using the previous question's subject)
- Starts with "و" (Arabic "and") and references a previous query implicitly

If the question is already fully self-contained (has a clear subject and analytical intent), return it UNCHANGED.
Return ONLY the rewritten question — one sentence, no explanation.
""")

# ── Natural language summary ──────────────────────────────────
NL_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["question", "data_preview", "columns"],
    template="""
You are a retail analytics assistant. Summarize the data result below in exactly 2-3 clear, concise sentences.
Reference specific numbers, entities, and patterns visible in the data.
Be direct and insightful — no filler phrases like "the table shows" or "as we can see".
CRITICAL LANGUAGE RULE: Detect the language of the question below. If it contains Arabic characters → write the ENTIRE summary in Arabic. If it is English → write entirely in English. NEVER mix languages. NEVER translate the question.
CURRENCY: use Kuwaiti Dinar (KWD / دينار كويتي) for any monetary value.

Question: {question}
Columns: {columns}
Data:
{data_preview}

Summary (2-3 sentences only):
""")

# ── Chart edit gate ───────────────────────────────────────────
CHART_EDIT_GATE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a classifier. Decide if the user input is a request to MODIFY the currently displayed chart
(e.g. change chart type, flip orientation, change colors, sort differently, add/remove labels,
zoom in on a region, switch to log scale, highlight a specific bar or line)
OR if it is a new data query that requires fetching new data.

User input: {question}

Respond ONLY with one word:  chart_edit   OR   new_query
""")

# ── Chart edit code generator ─────────────────────────────────
CHART_EDIT_PROMPT = PromptTemplate(
    input_variables=["instruction", "plotly_code", "columns", "data_preview"],
    template="""
You are a senior data visualization expert. The user wants to modify an existing chart.

Instruction: {instruction}
DataFrame columns available: {columns}
Data preview:
{data_preview}

Existing plotly code to modify:
{plotly_code}

Rewrite ONLY the plotly code to apply the requested modification.
- Keep all unchanged aspects of the chart identical.
- Apply fix_arabic() on any new Arabic text strings you add.
- Store the figure as `fig`. End with fig.show().
- fix_arabic() and the `result` DataFrame are already in memory.
Write ONLY raw Python/plotly code. No markdown, no fences, no comments.
""")

# ── Spell / typo correction ───────────────────────────────────
SPELL_CORRECT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are an Arabic/English text correction assistant.
Fix any obvious spelling or typographic mistakes in the following query.
Do NOT change numbers, names, entity names, or meaning — only fix clear misspellings.
If the text is already correct, return it UNCHANGED.
Respond with ONLY the corrected text — no explanation, no extra words.

Query: {question}

Corrected:""")

# ── Summary fallback ──────────────────────────────────────────
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["question", "error", "schema_hint"],
    template="""
You are a retail analytics expert. A technical query could not be executed automatically.

User question: {question}
Error: {error}
Schema hint: {schema_hint}

Provide a helpful TEXT-ONLY answer:
- Explain what the data analysis would typically show.
- Suggest how to rephrase the question for better results.
- If possible, give a rough qualitative answer based on general retail knowledge.

Respond in the SAME LANGUAGE as the question. Be concise (2-4 sentences).
""")
