# AI Data Science Copilot - Startup Script for Windows (PowerShell)

# 0. Set PYTHONPATH so Streamlit can find 'src'
$env:PYTHONPATH = "."

# 1. Activate the Python 3.11 virtual environment
Write-Host "[VENV] Activating virtual environment..." -ForegroundColor Cyan
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
} else {
    Write-Error "Virtual environment not found! Run 'uv venv --python 3.11' first."
    Exit 1
}

# 2. Start the FastAPI backend in the background
Write-Host "[API] Starting FastAPI Backend API on http://127.0.0.1:8000..." -ForegroundColor Green
$apiProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn src.api:app --host 127.0.0.1 --port 8000" -NoNewWindow -PassThru

# 3. Wait a moment for FastAPI to initialize
Start-Sleep -Seconds 3

# 4. Start the Streamlit frontend
Write-Host "[UI] Starting Streamlit Frontend Dashboard on http://127.0.0.1:8501..." -ForegroundColor Green
streamlit run src/app.py --server.port 8501

# Cleanup: Stop the background API process when Streamlit exits
if ($apiProcess) {
    Write-Host "[STOP] Shutting down backend API process..." -ForegroundColor Yellow
    Stop-Process -Id $apiProcess.Id -Force
}
