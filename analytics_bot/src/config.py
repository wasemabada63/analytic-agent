"""
config.py
=========
All paths, environment variables, data loading, FAISS index, and runtime
date constants.  Import this module FIRST — every other module depends on it.
"""
import os
import datetime
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine

# ── Environment ───────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_dwh_index")

EXPORT_XLSX  = os.path.join(BASE_DIR, "last_result.xlsx")
EXPORT_CSV   = os.path.join(BASE_DIR, "last_result.csv")
EXPORT_HTML  = os.path.join(BASE_DIR, "last_chart.html")
QUERY_LOG    = os.path.join(BASE_DIR, "query_log.jsonl")
SESSION_FILE = os.path.join(BASE_DIR, "session.json")

# Dummy definitions for removed legacy CSV mode, to avoid import errors elsewhere
orders = products = order_items = categories = subcategories = None

# ── DWH connection (SQL pipeline) ─────────────────────────────
engine = None
DB_ERROR = None
DWH_SCHEMA = os.getenv("DWH_SCHEMA", "dwh1")

try:
    DWH_USER   = os.getenv("DWH_USER", "")
    DWH_PASS   = os.getenv("DWH_PASS", "")
    DWH_HOST   = os.getenv("DWH_HOST", "")
    DWH_PORT   = os.getenv("DWH_PORT", "6543")
    DWH_NAME   = os.getenv("DWH_NAME", "postgres")
    DWH_STATEMENT_TIMEOUT_MS = int(os.getenv("DWH_STATEMENT_TIMEOUT_MS", "30000"))

    if not all([DWH_USER, DWH_PASS, DWH_HOST, DWH_NAME]):
        raise RuntimeError("Missing required DWH environment variables (.env).")

    engine = create_engine(
        f"postgresql+psycopg2://{DWH_USER}:{quote_plus(DWH_PASS)}"
        f"@{DWH_HOST}:{DWH_PORT}/{DWH_NAME}",
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={
            "options": f"-c statement_timeout={DWH_STATEMENT_TIMEOUT_MS}",
        },
    )
    with engine.connect() as conn:
        pass
    print(f"✅ DWH engine ready ({DWH_HOST}:{DWH_PORT}/{DWH_NAME} schema={DWH_SCHEMA}).")
except Exception as e:
    engine = None
    DB_ERROR = str(e)
    print(f"⚠️ DWH engine failed to connect: {e}")

# ── FAISS vector store ────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
try:
    vector_store = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    print(f"✅ FAISS index loaded ({os.path.basename(INDEX_DIR)}).")
except Exception as e:
    print(f"⚠️ FAISS index failed to load: {e}")
    vector_store = None

# ── Runtime date context ──────────────────────────────────────
TODAY = datetime.datetime.now().strftime("%B %Y")

DATA_MIN_DATE, DATA_MAX_DATE = "unknown", "unknown"
DATA_YEARS = []

if engine:
    try:
        _range = pd.read_sql_query(
            f"""
            SELECT MIN(d.full_date) AS min_d,
                   MAX(d.full_date) AS max_d,
                   ARRAY_AGG(DISTINCT d.year ORDER BY d.year) AS years
            FROM {DWH_SCHEMA}.fact_order_item f
            JOIN {DWH_SCHEMA}.dim_date d ON f.order_date_key = d.date_key
            WHERE d.full_date IS NOT NULL;
            """,
            engine,
        )
        DATA_MIN_DATE = _range["min_d"].iloc[0].strftime("%Y-%m-%d")
        DATA_MAX_DATE = _range["max_d"].iloc[0].strftime("%Y-%m-%d")
        DATA_YEARS    = list(_range["years"].iloc[0])
    except Exception as e:
        print(f"⚠️  Could not compute date range from DWH ({e}); using fallback.")
        DATA_MIN_DATE, DATA_MAX_DATE = "2023-12-28", "2025-08-01"
        DATA_YEARS = [2023, 2024, 2025]
else:
    DATA_MIN_DATE, DATA_MAX_DATE = "2023-12-28", "2025-08-01"
    DATA_YEARS = [2023, 2024, 2025]

DATA_YEAR_RANGE = f"{DATA_YEARS[0]}–{DATA_YEARS[-1]}" if DATA_YEARS else "unknown"
print(f"📅 Data range: {DATA_MIN_DATE} → {DATA_MAX_DATE} | Years: {DATA_YEARS}")
