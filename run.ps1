param(
    [int]$Port = 8501,
    [switch]$Headless
)

# Ensure we run from the script's directory
Set-Location -Path $PSScriptRoot

# Create venv if missing
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment (.venv)..."
    py -3.11 -m venv .venv
}

# Activate venv
. .\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements if needed
Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip | Out-Null
pip install -r requirements.txt

# Build streamlit args
$ArgsList = @("-m", "streamlit", "run", "steamlit.py")
if ($Headless) { $ArgsList += @("--server.headless", "true") }
if ($Port) { $ArgsList += @("--server.port", "$Port") }

# Run the app
Write-Host "Starting Streamlit on port $Port..." -ForegroundColor Green
python @ArgsList
