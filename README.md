# AI Data Science Copilot

An interactive data science assistant for exploring datasets, improving data quality, building machine-learning models, and generating explanations and reports. The project includes a Streamlit dashboard and a FastAPI backend.

## Features

- Upload CSV, Excel (`.xls`/`.xlsx`), and JSON datasets.
- Automatically profile datasets, suggest a target column and problem type, and calculate a dataset health score.
- Explore distributions and relationships with interactive visualizations.
- Review and apply data-cleaning and feature-engineering recommendations.
- Analyze bias and fairness, and generate synthetic or rebalanced data.
- Train and tune machine-learning models, make predictions, and inspect model explanations.
- Compare datasets for drift, chat with the assistant, and export PDF reports.

## Requirements

- Python 3.11
- Windows PowerShell (the included `run.ps1` startup script is for Windows)

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, allow scripts for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

With the virtual environment active, start the dashboard and API together:

```powershell
.\run.ps1
```

Open the Streamlit dashboard at <http://localhost:8501>. The FastAPI backend runs at <http://localhost:8000>; its interactive API documentation is available at <http://localhost:8000/docs>.

To start either service separately:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

```powershell
streamlit run src/app.py --server.port 8501
```

## Optional language-model integration

The assistant can use OpenAI when `OPENAI_API_KEY` is set in the environment. If no OpenAI key is available, language-model features can use a locally running [Ollama](https://ollama.com/) server. Set `OLLAMA_HOST` if Ollama is listening somewhere other than `http://localhost:11434`. The core data-analysis and machine-learning features do not require an LLM.

For example, set these for the current PowerShell session before launching the app:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OLLAMA_HOST = "http://localhost:11434"
```

Keep API keys private; do not commit them to the repository.

## Run the verification script

With the project dependencies installed:

```powershell
python tests/verify_engines.py
```
