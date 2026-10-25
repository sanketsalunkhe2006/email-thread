# Email Threat Detector — Glass UI

A local Streamlit learning project that classifies pasted email text as `phishing` or `legitimate`, highlights common warning patterns, and saves analysis history in SQLite.

## Setup on Windows / VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If PowerShell blocks virtual-environment activation, no activation is required when using the `.venv\Scripts\python.exe` commands above. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Dataset

`data/emails.csv` must contain `text` and `label` columns. Valid labels are `phishing` and `legitimate`. The included sample rows are synthetic educational examples and are not validated real-world email data.

## Features

- Glassmorphism UI with local AI/cybersecurity-themed SVG background
- Email subject and body analysis
- TF-IDF + Logistic Regression model
- Rule-based warning sign categories
- Persistent SQLite analysis history
- Search and CSV download for history
- Model metrics and Terms & Conditions pages
- Delete-history confirmation

## Privacy and limitations

Text is processed by the local Python application; this version does not send email text to an AI API. The local `email_history.db` file stores analysis metadata and a limited email text excerpt. Do not paste passwords, OTPs, confidential messages, or sensitive personal data. Anyone with access to the project folder may access the local history.

This is a moderate-sized educational prototype, not a production security product. It does not connect to mailboxes, inspect attachments, verify sender domains, follow URLs, or query live threat-intelligence services. Predictions can be wrong; always verify suspicious requests through a trusted channel.


## Layout troubleshooting

The app CSS uses a full-width main container with responsive padding. If the page looks narrow, maximize the browser, press `Ctrl+0` to reset zoom, and restart Streamlit.
