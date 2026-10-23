"""
NeoBank AI Studio — modular-ready single-file demo
Run:
    python -m pip install -r requirements.txt
    streamlit run app.py

Demo/simulation only. Do not use for real banking, real customer data, or real money.
"""
from __future__ import annotations

import os
import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("NEOBANK_DB_PATH", str(APP_DIR / "neobank_demo.db")))
MODEL_PATH = APP_DIR / "fraud_model.joblib"

st.set_page_config(page_title="NeoBank AI Studio", page_icon="🏦", layout="wide")

# ----------------------------- Persistence -----------------------------
def db_connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db_connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            account_type TEXT NOT NULL DEFAULT 'Savings',
            balance REAL NOT NULL DEFAULT 0 CHECK(balance >= 0),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount > 0),
            note TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(account_id) REFERENCES accounts(account_id)
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        count = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
        if count == 0:
            now = utcnow()
            demo = [
                ("Aarav Sharma", "+919800000001", "Savings", 25000.0, now),
                ("Priya Singh", "+919800000002", "Current", 78000.0, now),
                ("Kabir Verma", "+919800000003", "Student", 8500.0, now),
            ]
            con.executemany(
                "INSERT INTO accounts(name,phone,account_type,balance,created_at) VALUES(?,?,?,?,?)",
                demo,
            )
            con.execute(
                "INSERT INTO audit_log(action,details,created_at) VALUES(?,?,?)",
                ("DEMO_INIT", "Created sample accounts for demonstration", now),
            )

def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def audit(action: str, details: str):
    with db_connect() as con:
        con.execute(
            "INSERT INTO audit_log(action,details,created_at) VALUES(?,?,?)",
            (action, details[:500], utcnow()),
        )

def accounts_df():
    with db_connect() as con:
        return pd.read_sql_query(
            "SELECT account_id, name, phone, account_type, balance, created_at FROM accounts ORDER BY account_id DESC",
            con,
        )

def transactions_df(limit: int = 1000):
    with db_connect() as con:
        return pd.read_sql_query(
            """SELECT t.transaction_id, t.account_id, a.name, t.kind, t.amount, t.note, t.created_at
               FROM transactions t JOIN accounts a ON a.account_id=t.account_id
               ORDER BY t.transaction_id DESC LIMIT ?""",
            con, params=(limit,),
        )

def log_transaction(account_id: int, kind: str, amount: float, note: str = ""):
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("Amount must be a finite number greater than zero.")
    with db_connect() as con:
        row = con.execute("SELECT balance FROM accounts WHERE account_id=?", (account_id,)).fetchone()
        if row is None:
            raise ValueError("Account not found.")
        balance = float(row["balance"])
        if kind in ("WITHDRAW", "TRANSFER_OUT") and balance < amount:
            raise ValueError("Insufficient balance.")
        delta = amount if kind in ("DEPOSIT", "TRANSFER_IN") else -amount
        con.execute("UPDATE accounts SET balance=balance+? WHERE account_id=?", (delta, account_id))
        con.execute(
            "INSERT INTO transactions(account_id,kind,amount,note,created_at) VALUES(?,?,?,?,?)",
            (account_id, kind, float(amount), note[:250], utcnow()),
        )
        con.execute(
            "INSERT INTO audit_log(action,details,created_at) VALUES(?,?,?)",
            (kind, f"account_id={account_id}; amount={amount:.2f}; note={note[:100]}", utcnow()),
        )

def create_account(name: str, phone: str, account_type: str, opening: float):
    if not name.strip():
        raise ValueError("Name is required.")
    if opening < 0 or not np.isfinite(opening):
        raise ValueError("Opening balance cannot be negative.")
    with db_connect() as con:
        cur = con.execute(
            "INSERT INTO accounts(name,phone,account_type,balance,created_at) VALUES(?,?,?,?,?)",
            (name.strip(), phone.strip(), account_type, float(opening), utcnow()),
        )
        account_id = int(cur.lastrowid)
        if opening > 0:
            con.execute(
                "INSERT INTO transactions(account_id,kind,amount,note,created_at) VALUES(?,?,?,?,?)",
                (account_id, "OPENING", float(opening), "Opening balance", utcnow()),
            )
        con.execute(
            "INSERT INTO audit_log(action,details,created_at) VALUES(?,?,?)",
            ("CREATE_ACCOUNT", f"account_id={account_id}; name={name.strip()}", utcnow()),
        )
    return account_id

init_db()

# ----------------------------- ML helpers -----------------------------
def make_synthetic_training_data(n: int = 900, seed: int = 42):
    """Synthetic data for a teaching demo; not a validated fraud dataset."""
    rng = np.random.default_rng(seed)
    amounts = rng.lognormal(mean=5.1, sigma=1.0, size=n)
    hours = rng.integers(0, 24, size=n)
    recent_count = rng.poisson(2, size=n)
    distance_km = rng.exponential(12, size=n)
    # Synthetic labels only for demonstration.
    risk = (
        (amounts > 1800).astype(int)
        + (hours <= 4).astype(int)
        + (recent_count > 5).astype(int)
        + (distance_km > 45).astype(int)
    )
    y = (risk >= 2).astype(int)
    X = pd.DataFrame({
        "amount": amounts,
        "hour": hours,
        "recent_tx_count": recent_count,
        "distance_km": distance_km,
    })
    return X, y

@st.cache_resource
def train_demo_classifier():
    X, y = make_synthetic_training_data()
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=140, random_state=42, class_weight="balanced")),
    ])
    model.fit(X, y)
    return model

@st.cache_resource
def train_anomaly_model():
    X, _ = make_synthetic_training_data(n=1200, seed=11)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("detector", IsolationForest(n_estimators=150, contamination=0.04, random_state=42)),
    ])
    model.fit(X)
    return model

def transaction_features(amount: float, hour: int, recent_tx_count: int, distance_km: float):
    return pd.DataFrame([{
        "amount": float(amount),
        "hour": int(hour),
        "recent_tx_count": int(recent_tx_count),
        "distance_km": float(distance_km),
    }])

# ----------------------------- LLM helper -----------------------------
def get_llm_response(prompt: str, context: str) -> str:
    """
    Optional OpenAI-compatible API. Set OPENAI_API_KEY and optionally
    OPENAI_MODEL / OPENAI_BASE_URL in environment or Streamlit secrets.
    If no key is set, return a safe local FAQ response.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content":
                     "You are NeoBank Demo Assistant. Explain the demo app and general financial concepts. "
                     "Never claim to execute transactions. Never request passwords, PINs, OTPs, card details, "
                     "or secret credentials. Do not present yourself as a real bank or give guaranteed financial advice. "
                     "If asked about account-specific facts, use only the provided context. "
                     "State that this is a demo system, not a real banking service.\nContext:\n" + context[:10000]},
                    {"role": "user", "content": prompt[:3000]},
                ],
            )
            return response.choices[0].message.content or "I couldn't generate a response."
        except Exception as exc:
            return f"LLM connection failed ({type(exc).__name__}). Check your API key, model, and internet connection."
    q = prompt.lower()
    if any(word in q for word in ["fraud", "risk", "suspicious"]):
        return ("Fraud detection in this demo combines a synthetic supervised classifier and an Isolation Forest "
                "anomaly detector. Their scores are educational examples, not proof of fraud. Review unusual "
                "activity using verified records and a human review process.")
    if any(word in q for word in ["balance", "account", "customer"]):
        return "I can help explain the demo's account dashboard and reports. For privacy, this assistant only receives the limited context shown to it."
    if any(word in q for word in ["machine learning", "ml", "model"]):
        return "Machine learning finds patterns in example data. This demo uses Random Forest for a synthetic risk label, Isolation Forest for anomaly detection, and K-means for customer segmentation."
    return ("I'm the local NeoBank demo assistant. Ask about the dashboard, synthetic fraud scoring, customer segments, "
            "or how the Streamlit project is structured. Configure an LLM API key to enable generated answers.")

# ----------------------------- UI -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(100,120,150,.08); border: 1px solid rgba(120,130,150,.2); padding: 14px; border-radius: 14px;}
.hero {padding: 22px 24px; border-radius: 18px; background: linear-gradient(120deg,#101b36,#164e63); color: white; margin-bottom: 18px;}
.hero h1 {margin:0; font-size:2rem;} .hero p {margin:.4rem 0 0 0; opacity:.86;}
.small-note {font-size:.85rem; opacity:.75;}
</style>
<div class="hero"><h1>🏦 NeoBank AI Studio</h1>
<p>Python · Streamlit · Machine Learning · LLM assistant · Analytics</p></div>
""", unsafe_allow_html=True)
st.warning("Educational demo only. It is not a real banking system and must not process real money or sensitive customer data.")

pages = [
    "Overview", "Accounts", "Transactions", "AI Fraud Lab",
    "Customer Segmentation", "LLM Assistant", "Audit & Export", "Project Guide"
]
page = st.sidebar.radio("Navigation", pages)
st.sidebar.caption("Storage: local SQLite database")
st.sidebar.caption(f"Database file: {DB_PATH.name}")

accounts = accounts_df()
tx = transactions_df()

if page == "Overview":
    st.subheader("Executive overview")
    total_balance = float(accounts["balance"].sum()) if not accounts.empty else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo accounts", f"{len(accounts):,}")
    c2.metric("Total demo balance", f"₹{total_balance:,.2f}")
    c3.metric("Transactions", f"{len(tx):,}")
    c4.metric("Risk lab", "Ready", help="Uses synthetic training data")
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Balance by account")
        if not accounts.empty:
            st.bar_chart(accounts.set_index("name")["balance"])
        else:
            st.info("Create an account to see the chart.")
    with right:
        st.markdown("#### Transaction activity")
        if not tx.empty:
            daily = tx.assign(day=pd.to_datetime(tx["created_at"], errors="coerce").dt.date)
            daily = daily.groupby("day").size().rename("transactions")
            st.line_chart(daily)
        else:
            st.info("No transactions yet.")
    st.markdown("#### Latest transactions")
    st.dataframe(tx.head(10), use_container_width=True, hide_index=True)

elif page == "Accounts":
    st.subheader("Account management")
    tab1, tab2 = st.tabs(["Create account", "Account directory"])
    with tab1:
        with st.form("create_account_form", clear_on_submit=True):
            name = st.text_input("Customer name")
            phone = st.text_input("Phone (demo data only)", placeholder="+91...")
            account_type = st.selectbox("Account type", ["Savings", "Current", "Salary", "Student"])
            opening = st.number_input("Opening balance (₹)", min_value=0.0, value=0.0, step=500.0)
            submit = st.form_submit_button("Create account", type="primary")
        if submit:
            try:
                new_id = create_account(name, phone, account_type, opening)
                st.success(f"Demo account created. Account ID: {new_id}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with tab2:
        st.dataframe(accounts, use_container_width=True, hide_index=True)
        if not accounts.empty:
            chosen = st.selectbox("Choose account to view", accounts["account_id"].tolist(), format_func=lambda x: f"ID {x} — {accounts.loc[accounts.account_id == x, 'name'].iloc[0]}")
            row = accounts.loc[accounts.account_id == chosen].iloc[0]
            st.info(f"Account {chosen} · {row['account_type']} · Balance ₹{row['balance']:,.2f}")

elif page == "Transactions":
    st.subheader("Deposit / withdrawal simulator")
    st.caption("This simulator changes only the local demo database. No real payments are made.")
    if accounts.empty:
        st.info("Create an account first.")
    else:
        with st.form("transaction_form"):
            options = accounts["account_id"].tolist()
            account_id = st.selectbox("Account", options, format_func=lambda x: f"ID {x} — {accounts.loc[accounts.account_id == x, 'name'].iloc[0]}")
            kind = st.selectbox("Operation", ["DEPOSIT", "WITHDRAW"])
            amount = st.number_input("Amount (₹)", min_value=1.0, value=500.0, step=100.0)
            note = st.text_input("Note", max_chars=250)
            go = st.form_submit_button("Submit demo transaction", type="primary")
        if go:
            try:
                log_transaction(int(account_id), kind, float(amount), note)
                st.success(f"{kind.title()} recorded in the demo database.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.markdown("#### Transaction history")
    st.dataframe(tx, use_container_width=True, hide_index=True)

elif page == "AI Fraud Lab":
    st.subheader("AI transaction risk lab")
    st.write("Enter example transaction characteristics. The model is trained on synthetic data created in this file.")
    model = train_demo_classifier()
    anomaly = train_anomaly_model()
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Transaction amount (₹)", min_value=1.0, value=2500.0, step=100.0)
        hour = st.slider("Hour of day", 0, 23, 2)
    with c2:
        recent = st.slider("Transactions in recent period", 0, 20, 7)
        distance = st.number_input("Distance from usual location (km)", min_value=0.0, value=60.0, step=5.0)
    if st.button("Analyze transaction", type="primary"):
        features = transaction_features(amount, hour, recent, distance)
        probability = float(model.predict_proba(features)[0, 1])
        anomaly_label = int(anomaly.predict(features)[0])
        anomaly_score = float(-anomaly.decision_function(features)[0])
        col1, col2, col3 = st.columns(3)
        col1.metric("Synthetic risk score", f"{probability:.1%}")
        col2.metric("Anomaly result", "Unusual" if anomaly_label == -1 else "Typical")
        col3.metric("Anomaly score", f"{anomaly_score:.3f}")
        if probability >= 0.65 or anomaly_label == -1:
            st.warning("Review recommended. This output is a demo model score, not a fraud determination.")
        else:
            st.success("The demo model did not identify a high-risk pattern. This is not a guarantee of safety.")
        audit("AI_RISK_CHECK", f"amount={amount}; hour={hour}; recent={recent}; distance={distance}; p={probability:.4f}; anomaly={anomaly_label}")
    st.info("Production systems need representative, labelled data, calibration, bias testing, monitoring, access control, and human review. Never use these synthetic scores to block real customers.")

elif page == "Customer Segmentation":
    st.subheader("Customer segmentation (K-means)")
    if len(accounts) < 3:
        st.info("Create at least three accounts to run a simple demo segmentation.")
    else:
        features = accounts[["balance"]].copy()
        features["transaction_count"] = features["account_id"].map(tx["account_id"].value_counts()).fillna(0)
        scaler = StandardScaler()
        scaled = scaler.featuresfit_transform(features[["balance", "transaction_count"]])
        k = st.slider("Number of segments", 2, min(5, len(accounts)), min(3, len(accounts)))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        features["segment"] = model.fit_predict(scaled).astype(str)
        result = accounts[["account_id", "name", "account_type", "balance"]].merge(
            features[["account_id", "transaction_count", "segment"]], on="account_id"
        )
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.bar_chart(result.groupby("segment")[["balance", "transaction_count"]].mean())
        st.caption("Segments are exploratory groups, not credit ratings or recommendations.")

elif page == "LLM Assistant":
    st.subheader("NeoBank conversational assistant")
    st.caption("Use this to ask about app features or general ML concepts. Never enter PINs, passwords, OTPs, or real customer secrets.")
    context_accounts = accounts[["account_id", "account_type", "balance"]].head(20).to_dict(orient="records") if not accounts.empty else []
    context = json.dumps({
        "app": "NeoBank AI Studio educational demo",
        "account_count": len(accounts),
        "transaction_count": len(tx),
        "account_summary_without_names_or_phones": context_accounts,
        "features": ["synthetic fraud scoring", "Isolation Forest", "K-means segmentation", "SQLite demo storage"],
    })
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about AI fraud detection, Streamlit, ML, or this demo...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = get_llm_response(prompt, context)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    with st.expander("Configure an LLM API"):
        st.write("Set `OPENAI_API_KEY` as an environment variable or Streamlit secret. Do not put API keys directly in source code.")
        st.code('OPENAI_API_KEY = "your-key-in-environment-or-secrets"\nOPENAI_MODEL = "gpt-4.1-mini"', language="toml")
        st.write("Without a key, the app uses a small local FAQ fallback and does not call an LLM.")

elif page == "Audit & Export":
    st.subheader("Audit log and data export")
    with db_connect() as con:
        logs = pd.read_sql_query("SELECT log_id, action, details, created_at FROM audit_log ORDER BY log_id DESC LIMIT 1000", con)
    st.dataframe(logs, use_container_width=True, hide_index=True)
    st.download_button("Download accounts CSV", accounts.to_csv(index=False).encode("utf-8"), "neobank_accounts_demo.csv", "text/csv")
    st.download_button("Download transactions CSV", tx.to_csv(index=False).encode("utf-8"), "neobank_transactions_demo.csv", "text/csv")
    st.download_button("Download audit log CSV", logs.to_csv(index=False).encode("utf-8"), "neobank_audit_demo.csv", "text/csv")
    st.caption("Exports contain demo data. Protect exports carefully in any real system.")

elif page == "Project Guide":
    st.subheader("Project architecture and next steps")
    st.markdown("""
    **Current demo**
    - Streamlit UI with navigation and dashboard charts.
    - SQLite persistence on the machine running the app.
    - Account creation and deposit/withdraw simulation.
    - Synthetic Random Forest risk classifier plus Isolation Forest anomaly detection.
    - K-means customer grouping.
    - Optional LLM API integration with a local FAQ fallback.
    - Audit log and CSV export.

    **Recommended production-grade next steps**
    1. Split services, UI, models, and data-access code into separate modules.
    2. Add authentication, role-based permissions, CSRF/session protections, and rate limits.
    3. Use a managed database with migrations, backups, encryption, and least-privilege credentials.
    4. Add tests for transfers, concurrent updates, invalid input, and access-control rules.
    5. Build a RAG pipeline for approved policy documents and cite retrieved passages.
    6. Train and evaluate fraud models on properly governed, representative labelled data.
    7. Use a payment provider sandbox for simulated payment flows; never store card PINs or CVVs.
    8. Deploy secrets through platform secrets, not in Git.

    **Hosting note:** Streamlit Community Cloud local files are not guaranteed to be durable. This SQLite demo is intended for learning/local use; use a managed database for persistent hosted data.
    """)
    st.code("""neobank-ai-studio/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
""", language="text")

st.divider()
st.caption("NeoBank AI Studio · Learning prototype · No real financial transactions · Do not enter sensitive personal data")
