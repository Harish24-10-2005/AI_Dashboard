# 🚀 AI Dashboard Generator

This Streamlit app analyzes and merges datasets, optionally adds AI summaries and AI-driven visualizations, and renders a dashboard.

## Quick start (Windows PowerShell)

Run these commands from the project root (e.g., `D:\Ai_Dashboard`).

```powershell
# Create venv (Python 3.11) and activate
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the app
python -m streamlit run steamlit.py
```

Or use the convenience script:

```powershell
# Default port: 8501
./run.ps1

# Headless on port 8502 (useful on servers)
./run.ps1 -Headless -Port 8502
```

## Provide datasets in the sidebar

Enter a name and file path for each dataset. Sample files included in the repo:

- `relation_data\employees.csv`
- `relation_data\departments.csv`
- `relation_data\employee_projects.json`
- `relation_data\datasets.xlsx`

## Optional: enable AI features (Cohere)

Set your API key (open a new terminal after setting):

```powershell
setx COHERE_API_KEY "your_real_key_here"
```

If no key is set, the app runs normally but skips AI summaries and AI visualizations.

## Optional: configure Kaggle

Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json` to enable loading `kaggle:` sources.

Example sidebar path:

```
kaggle:username/dataset-name
```

## Troubleshooting

- Port busy → run `./run.ps1 -Port 8502`
- Missing data → ensure the provided file paths exist and are readable
- AI disabled → set `COHERE_API_KEY` and restart the terminal


