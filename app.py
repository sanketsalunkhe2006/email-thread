from __future__ import annotations

import base64
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Paths and Streamlit setup
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "email_model.joblib"
METRICS_FILE = ROOT / "models" / "metrics.json"
DB_FILE = ROOT / "email_history.db"
BG_FILE = ROOT / "assets" / "ai_security_background.svg"

st.set_page_config(
    page_title="Email Threat Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Glassmorphism CSS. Keep all CSS inside this string so it is never displayed
# as ordinary page text.
# -----------------------------------------------------------------------------
_bg_data = ""
if BG_FILE.exists():
    try:
        _bg_data = base64.b64encode(BG_FILE.read_bytes()).decode("ascii")
    except OSError:
        _bg_data = ""

_background = (
    f'url("data:image/svg+xml;base64,{_bg_data}") center / cover fixed'
    if _bg_data
    else "linear-gradient(135deg, #071329, #101f42)"
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {{ --muted: #aec2e4; --border: rgba(128,176,255,.28); }}
    html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
    .stApp {{
        color: #edf4ff;
        background: linear-gradient(110deg, rgba(3,9,25,.93), rgba(5,17,42,.84)), {_background};
    }}
    [data-testid="stHeader"] {{ background: rgba(4,12,31,.25); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(12,29,64,.97), rgba(5,14,34,.98));
        border-right: 1px solid var(--border);
    }}
    .block-container {{
        width: 100% !important; max-width: 1600px !important;
        padding: 1.5rem clamp(1rem, 3vw, 2.5rem) 3rem !important;
        margin-left: auto !important; margin-right: auto !important;
        box-sizing: border-box !important;
    }}
    [data-testid="stAppViewContainer"] {{ width: 100%; }}
    .block-container {{ box-sizing: border-box; }}
    h1, h2, h3, h4 {{ color: #f5f8ff !important; letter-spacing: -.025em; overflow-wrap: anywhere; }}
    [data-testid="stAlert"] {{ border-radius: 12px; }}
    p, label, .stMarkdown {{ color: #e3ecff; }}
    .hero {{
        padding: 1.5rem 1.7rem; border-radius: 22px;
        background: linear-gradient(115deg, rgba(22,68,133,.82), rgba(25,40,83,.68));
        border: 1px solid var(--border); box-shadow: 0 18px 50px rgba(0,0,0,.2);
        backdrop-filter: blur(18px); margin-bottom: 1.1rem;
    }}
    .eyebrow {{ color: #a9c9ff; font-size: .78rem; font-weight: 700; letter-spacing: .13em; }}
    .glass {{
        border: 1px solid var(--border); border-radius: 18px;
        background: linear-gradient(145deg, rgba(18,43,84,.83), rgba(9,23,52,.78));
        box-shadow: 0 12px 32px rgba(0,0,0,.18); padding: 1.1rem 1.25rem;
        backdrop-filter: blur(16px); min-height: 0; height: auto;
        box-sizing: border-box; overflow-wrap: anywhere; word-break: normal;
    }}
    .content-card, .chart-card {{ width: 100%; max-width: 100%; box-sizing: border-box; }}
    .content-card h3, .chart-card h3 {{ margin: 0 0 .75rem 0; font-size: 1.15rem; }}
    .content-card p {{ margin: .35rem 0; line-height: 1.55; }}
    [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
    [data-testid="stColumn"] {{ min-width: 0 !important; }}
    [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stAlert"] {{ max-width: 100%; }}
    [data-testid="stDataFrame"] {{ overflow: auto; }}
    .metric-label {{ color: #b8c9e8; font-size: .86rem; }}
    .metric-value {{ font-size: 1.85rem; font-weight: 800; color: #f7faff; margin-top: .15rem; }}
    .metric-note {{ color: #a9bddf; font-size: .78rem; margin-top: .2rem; }}
    .stTextInput input, .stTextArea textarea {{
        color: #f3f7ff !important; background: rgba(6,18,43,.86) !important;
        border: 1px solid rgba(122,171,255,.4) !important; border-radius: 12px !important;
    }}
    div[data-testid="stMetric"], div[data-testid="stExpander"],
    div[data-testid="stDataFrame"], div[data-testid="stFileUploader"] {{
        border: 1px solid var(--border); background: rgba(12,31,66,.62); border-radius: 15px;
    }}
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
        border: 1px solid #7cb0ff77; border-radius: 11px; color: white;
        background: linear-gradient(100deg, #1976f3, #5c5cf5); font-weight: 600;
        min-height: 2.65rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: #a9c9ff; color: white; filter: brightness(1.08);
    }}
    hr {{ border-color: #8bb0ec33; }}
    footer, #MainMenu {{ visibility: hidden; }}
    @media (max-width: 768px) {{
        .block-container {{ padding: 1rem .75rem 2rem !important; }}
        .hero {{ padding: 1rem; border-radius: 16px; }}
        .glass {{ min-width: 0; padding: .85rem; }}
        div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: .8rem; }}
        div[data-testid="stHorizontalBlock"] > div {{ min-width: min(100%, 240px) !important; flex: 1 1 100% !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Database and model helpers
# -----------------------------------------------------------------------------
def init_db() -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_FILE)) as con:
        with con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    subject TEXT,
                    prediction TEXT NOT NULL,
                    phishing_score REAL NOT NULL,
                    warning_count INTEGER NOT NULL,
                    email_text TEXT
                )"""
            )


def save_result(result: dict) -> None:
    with closing(sqlite3.connect(DB_FILE)) as con:
        with con:
            con.execute(
                """INSERT INTO analysis_history
                (timestamp, subject, prediction, phishing_score, warning_count, email_text)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    result["subject"],
                    result["prediction"],
                    round(float(result["score"]) * 100, 2),
                    len(result["findings"]),
                    result["text"][:5000],
                ),
            )


def get_history() -> pd.DataFrame:
    with closing(sqlite3.connect(DB_FILE)) as con:
        return pd.read_sql_query(
            """SELECT id AS ID, timestamp AS Time, subject AS Subject,
            prediction AS Prediction, phishing_score AS "Phishing score (%)",
            warning_count AS "Warning signs"
            FROM analysis_history ORDER BY id DESC""",
            con,
        )


def clear_history() -> None:
    with closing(sqlite3.connect(DB_FILE)) as con:
        with con:
            con.execute("DELETE FROM analysis_history")


def get_model():
    if not MODEL_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        st.sidebar.error(f"The saved model could not be loaded: {exc}")
        return None


def get_metrics() -> dict:
    try:
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


RULES = {
    "Sensitive information request": [
        r"\bpassword\b", r"\botp\b", r"\bpin\b", r"\brecovery code\b",
        r"\bcard details\b", r"\bsecurity code\b", r"\blogin code\b",
    ],
    "Urgency or pressure": [
        r"\bimmediately\b", r"\burgent\b", r"\bact now\b",
        r"\bwithin \d+ minutes?\b", r"\bwill be (blocked|closed|deleted|suspended)\b",
        r"\bfinal warning\b",
    ],
    "Unexpected money or prize": [
        r"\badvance fee\b",
        r"\bgift cards?\b",
        r"\blottery\b",
        r"\bpay\b.{0,60}\b(release|unlock|claim)\b",
        r"\btransfer your funds\b",
        r"\bclaim your prize\b",
    ],
    "Suspicious link or attachment": [
        r"https?://\S+", r"\bshortened url\b", r"\bunknown attachment\b",
        r"\benable macros\b", r"\bdisable antivirus\b", r"\bremote access\b",
    ],
}


def find_warnings(email_text: str) -> list[str]:
    return [
        category
        for category, patterns in RULES.items()
        if any(re.search(pattern, email_text, re.IGNORECASE) for pattern in patterns)
    ]


def predict_email(email_text: str, model) -> tuple[str, float]:
    """Return a classification and estimated phishing probability.

    If no trained model is available, use a clearly labelled rules-based fallback.
    The fallback is educational and must not be treated as a security guarantee.
    """
    if model is not None:
        try:
            raw_label = str(model.predict([email_text])[0]).strip().lower()
            classes = [
                str(item).strip().lower()
                for item in getattr(model, "classes_", [])
            ]

            if hasattr(model, "predict_proba") and classes:
                probabilities = model.predict_proba([email_text])[0]
                if len(probabilities) == len(classes) and "phishing" in classes:
                    score = float(probabilities[classes.index("phishing")])
                else:
                    score = 1.0 if raw_label == "phishing" else 0.0
            else:
                score = 1.0 if raw_label == "phishing" else 0.0

            label = raw_label
            if label not in {"phishing", "legitimate"}:
                label = "phishing" if score >= 0.5 else "legitimate"

            return label, max(0.0, min(1.0, score))
        except (ValueError, TypeError, AttributeError, IndexError) as exc:
            raise RuntimeError(
                "The saved model could not classify this email. "
                "Check that it was created by train_model.py and uses "
                "'phishing' and 'legitimate' labels."
            ) from exc

    # Rules-only fallback: a basic demonstration, not a machine-learning result.
    findings = find_warnings(email_text)
    normalized = email_text.casefold()
    score = min(0.95, 0.12 + 0.22 * len(findings))

    high_risk_phrases = (
        "send your password",
        "share your otp",
        "reply with your password",
        "gift card codes",
        "send me your otp",
        "provide your login code",
    )
    if any(phrase in normalized for phrase in high_risk_phrases):
        score = max(score, 0.85)

    return ("phishing" if score >= 0.5 else "legitimate"), score


def hero(title: str, description: str, eyebrow: str = "EMAIL THREAT DETECTOR") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str | int, note: str, color: str = "#f7faff") -> None:
    st.markdown(
        f'<div class="glass"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Initialize application state
# -----------------------------------------------------------------------------
init_db()
model = get_model()
metrics = get_metrics()
history = get_history()

with st.sidebar:
    st.markdown("## Email Threat Detector")
    st.caption("AI-assisted email security")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Email", "Model Performance", "Analysis History", "Terms & Conditions", "About Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.write("Model status")
    if model is not None:
        st.success("Machine-learning model loaded")
    else:
        st.warning("Model not trained. Rules-only fallback is active.")
        st.caption("Run train_model.py to enable machine-learning predictions.")
    st.caption("Educational prototype. Verify suspicious messages through official channels.")

# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
if page == "Dashboard":
    hero("Email Threat Detection", "Inspect email text for phishing patterns and suspicious social-engineering signals.", "AI-ASSISTED CYBERSECURITY")
    total = len(history)
    phishing_count = int((history["Prediction"] == "phishing").sum()) if total else 0
    legitimate_count = int((history["Prediction"] == "legitimate").sum()) if total else 0
    accuracy = metrics.get("accuracy")
    accuracy_text = f"{float(accuracy) * 100:.1f}%" if isinstance(accuracy, (int, float)) else "N/A"

    columns = st.columns(4)
    cards = [
        ("Emails analyzed", str(total), "Saved on this device", "#72aaff"),
        ("Flagged as phishing", str(phishing_count), "Model/rules result", "#ff8fa3"),
        ("Predicted legitimate", str(legitimate_count), "Still verify messages", "#55e2be"),
        ("Test accuracy", accuracy_text, "Held-out sample data", "#b4a0ff"),
    ]
    for col, values in zip(columns, cards):
        with col:
            metric_card(*values)

    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        status_title = "System status"
        status_message = (
            "Machine-learning model is loaded."
            if model is not None
            else "The app is running in rules-only fallback mode."
        )
        st.markdown(
            f'<section class="glass content-card"><h3>{status_title}</h3>'
            f'<p>{status_message}</p>'
            '<p class="metric-note">This prototype cannot guarantee whether an email is safe.</p></section>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<section class="glass chart-card"><h3>Prediction distribution</h3></section>', unsafe_allow_html=True)
        if total:
            chart_data = pd.DataFrame(
                {"Classification": ["Phishing", "Legitimate"], "Count": [phishing_count, legitimate_count]}
            ).set_index("Classification")
            st.bar_chart(chart_data, use_container_width=True)
        else:
            st.info("Your chart appears after the first analysis.")

    st.write("")
    st.subheader("Recent analysis")
    if total:
        st.dataframe(history.drop(columns=["ID"]).head(8), use_container_width=True, hide_index=True)
    else:
        st.info("No saved analyses yet. Open Analyze Email to begin.")

# -----------------------------------------------------------------------------
# Analyze Email
# -----------------------------------------------------------------------------
elif page == "Analyze Email":
    hero("Analyze an email", "Enter an email subject and body. Avoid including real passwords, OTPs, or private information.")
    with st.form("email_analysis_form", clear_on_submit=False):
        subject = st.text_input("Email subject (optional)", placeholder="Urgent account verification")
        body = st.text_area("Email body", height=220, placeholder="Paste the email text you want to inspect...")
        submitted = st.form_submit_button("Analyze email", use_container_width=True)

    if submitted:
        if not (subject.strip() or body.strip()):
            st.warning("Enter an email subject or body first.")
        elif len(subject) + len(body) > 20000:
            st.error("Keep the combined subject and body under 20,000 characters.")
        else:
            email_text = (f"Subject: {subject.strip()}\n" if subject.strip() else "") + body.strip()
            with st.spinner("Checking email text..."):
                try:
                    label, score = predict_email(email_text, model)
                    findings = find_warnings(email_text)
                    result = {
                        "text": email_text,
                        "subject": subject.strip() or "(No subject)",
                        "prediction": label,
                        "score": score,
                        "findings": findings,
                    }
                    save_result(result)
                    st.session_state["last_email_result"] = result
                except Exception as exc:
                    st.error(str(exc))
                    st.caption(
                        "If this continues, retrain the model with train_model.py "
                        "and check the terminal for the full error."
                    )

    result = st.session_state.get("last_email_result")
    if result:
        st.divider()
        if result["prediction"] == "phishing":
            st.error("Result: Potential phishing — verify through an official channel.")
        else:
            st.success("Result: Likely legitimate — this does not guarantee safety.")
        col1, col2 = st.columns(2)
        col1.metric("Estimated phishing score", f"{result['score'] * 100:.1f}%")
        col2.metric("Warning categories", len(result["findings"]))
        if model is None:
            st.info("Rules-only estimate: train the machine-learning model for ML-based predictions.")
        else:
            st.caption("The score is a model estimate, not a guarantee of safety or malicious intent.")

        st.subheader("Warning sign review")
        if result["findings"]:
            for finding in result["findings"]:
                st.markdown(f"**{finding}** — A pattern associated with this warning category was detected.")
        else:
            st.info("No listed warning patterns were detected. This does not prove the email is safe.")
        st.subheader("Recommended next steps")
        st.markdown(
            "- Do not click unexpected links or open unexpected attachments.\n"
            "- Never share passwords, PINs, OTPs, or recovery codes by email.\n"
            "- Verify payment and account requests through a known official channel.\n"
            "- Report suspicious messages to your email provider or organization."
        )

    st.divider()
    st.subheader("Sample emails")
    sample_left, sample_right = st.columns(2)
    with sample_left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Routine meeting email**")
        st.write("The team meeting is scheduled for tomorrow at 10 AM. Please review the agenda in the official workspace.")
        st.markdown("</div>", unsafe_allow_html=True)
    with sample_right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Suspicious account email**")
        st.write("Your account will be suspended today. Reply immediately with your password and OTP.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model Performance
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    hero("Model performance", "Metrics from the held-out test split used during training.")
    if not metrics:
        st.info("No metrics found. Run train_model.py to train the model and create metrics.json.")
    else:
        metric_columns = st.columns(4)
        for col, (label, key) in zip(metric_columns, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1_score")]):
            value = metrics.get(key)
            with col:
                st.metric(label, f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "N/A")
        st.write(f"Training examples: {metrics.get('train_rows', 'N/A')}")
        st.write(f"Test examples: {metrics.get('test_rows', 'N/A')}")
        st.warning("The included dataset is synthetic teaching data. These scores do not estimate real-world security performance.")

# -----------------------------------------------------------------------------
# Analysis History
# -----------------------------------------------------------------------------
elif page == "Analysis History":
    hero("Analysis history", "Records are saved locally and remain after restarting the app.")
    history = get_history()
    if history.empty:
        st.info("No saved analysis history yet.")
    else:
        search_text = st.text_input("Search by subject or classification")
        filtered = history.copy()
        if search_text.strip():
            mask = (
                filtered["Subject"].fillna("").str.contains(search_text, case=False, regex=False)
                | filtered["Prediction"].str.contains(search_text, case=False, regex=False)
            )
            filtered = filtered[mask]
        st.dataframe(filtered.drop(columns=["ID"]), use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV report",
            data=filtered.drop(columns=["ID"]).to_csv(index=False).encode("utf-8"),
            file_name="email_analysis_history.csv",
            mime="text/csv",
        )
    with st.expander("Delete saved history"):
        st.warning("This permanently removes all saved analysis records.")
        confirmed = st.checkbox("I understand this cannot be undone.")
        if st.button("Delete all history", disabled=not confirmed):
            clear_history()
            st.session_state.pop("last_email_result", None)
            st.success("History deleted.")
            st.rerun()

# -----------------------------------------------------------------------------
# Terms and Conditions
# -----------------------------------------------------------------------------
elif page == "Terms & Conditions":
    hero("Terms and conditions", "Read these conditions before using the application.")
    terms = [
        ("Purpose", "This application is an educational prototype for learning about machine-learning-based email threat detection."),
        ("No security guarantee", "Predictions can be wrong. A legitimate result does not prove an email is safe, and a phishing result does not prove malicious intent."),
        ("Privacy", "Email text is processed by this local app. Analysis metadata and a limited text excerpt are stored in a local SQLite database. Do not enter sensitive or confidential messages."),
        ("Acceptable use", "Use the tool only for lawful, authorized, defensive, educational, or research purposes. Do not use it to facilitate phishing, fraud, harassment, or unauthorized access."),
        ("Data quality", "The included dataset is synthetic and limited. Model metrics may not reflect real-world performance."),
        ("User responsibility", "Verify suspicious requests through official channels. Do not click links, open attachments, or share credentials based only on this application's output."),
        ("Limitations", "This is not an enterprise security product. It does not scan mailboxes, inspect attachments, or check live threat-intelligence feeds."),
        ("Changes", "Project features and these terms may change as the project is updated."),
    ]
    for title, body_text in terms:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body_text)
    st.checkbox("I have read and understood these terms and conditions.")

# -----------------------------------------------------------------------------
# About Project
# -----------------------------------------------------------------------------
elif page == "About Project":
    hero("About this project", "A beginner-friendly machine-learning demonstration for defensive email security.")
    st.markdown("### Technology stack")
    st.write("Python, Streamlit, scikit-learn, TF-IDF, Logistic Regression, pandas, and SQLite.")
    st.markdown("### Project structure")
    st.code(
        "app.py                     Web interface\n"
        "train_model.py             Train and evaluate model\n"
        "requirements.txt           Dependencies\n"
        "data/emails.csv            Labeled sample dataset\n"
        "models/email_model.joblib  Trained model\n"
        "models/metrics.json        Test metrics\n"
        "email_history.db           Local saved history",
        language="text",
    )
    st.markdown("### Limitations")
    st.write("This local prototype classifies pasted text only. It does not connect to mailboxes, scan attachments, verify sender domains, follow links, or query live threat intelligence.")
