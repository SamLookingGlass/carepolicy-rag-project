# One-command demo: ingest (if needed) + start API
param(
    [switch]$SkipIngest,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "Virtual env not found. Run: py -3.13 -m venv .venv; pip install -e '.[dev]'"
}

if (-not $SkipIngest) {
    if (-not (Test-Path "data\processed\chunks.jsonl")) {
        Write-Host "Running first-time ingestion..."
        .\.venv\Scripts\python scripts\ingest_all.py
    } else {
        Write-Host "Corpus found (data\processed\chunks.jsonl). Skipping ingest. Use -SkipIngest:`$false and delete data\ to re-ingest."
    }
}

Write-Host "Starting CarePolicy RAG API on http://localhost:$Port"
Write-Host "Docs: http://localhost:$Port/docs"
.\.venv\Scripts\uvicorn src.api.main:app --host 0.0.0.0 --port $Port --reload
