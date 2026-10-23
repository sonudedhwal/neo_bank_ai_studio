# NeoBank AI Studio

Educational single-file Streamlit demo combining a banking-style dashboard, SQLite demo storage, synthetic ML risk scoring, K-means customer segmentation, an optional LLM assistant, audit logs, and CSV exports.

## Run on Windows / VS Code

```powershell
cd path\to\neobank-ai-studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Optional LLM configuration

Set environment variables in PowerShell before running:

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_MODEL="gpt-4.1-mini"
streamlit run app.py
```

Do not commit keys to GitHub. You can run without a key; the assistant uses a basic local FAQ fallback.

## Important limitations

- This is a learning/demo project, not a real bank.
- Fraud training data is synthetic and must not be used for real decisions.
- SQLite data lives on the machine/container running Streamlit. On hosted platforms it may not persist reliably across rebuilds.
- No authentication, real payment rails, production security, or regulatory controls are implemented.
- Do not enter real personal, account, payment, PIN, password, OTP, or card data.
