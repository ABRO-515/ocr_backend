# OCR Backend (FastAPI + OpenCV + PaddleOCR)

A production-grade FastAPI backend that extracts text from CNICs and vehicle plates using OpenCV preprocessing and PaddleOCR. Includes PostgreSQL, Adminer, Docker, and structured logging via structlog. Managed with the uv package manager. Compatible with Python 3.12.

## Features
- /upload-cnic: Extracts CNIC fields and stores them in PostgreSQL
- /upload-vehicle: Extracts vehicle number and stores it in PostgreSQL
- OpenCV preprocessing (resize, denoise, threshold)
- PaddleOCR text extraction
- structlog structured logging (JSON by default)
- Config via `config/settings.toml`
- Dockerized with `docker-compose` (Postgres + Adminer + API)

## Quickstart (Docker)
```bash
# Build and run all services
docker compose up -d --build

# API: http://localhost:8000/docs
# Adminer: http://localhost:8080 (System: PostgreSQL, Server: db, User: postgres, Password: postgres, DB: ocr_db)
```

## Local Dev (uv)
```bash
# Install uv (if not installed)
# Windows PowerShell:
# iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

uv venv --python 3.12
. .venv/Scripts/Activate.ps1  # on Windows
uv pip install -e .
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration
Edit `config/settings.toml`:
```toml
[app]
name = "OCR Backend"
environment = "development"
host = "0.0.0.0"
port = 8000

[logging]
level = "INFO"
json = true

[storage]
upload_dir = "uploads"

[database]
user = "postgres"
password = "postgres"
host = "db"
port = 5432
name = "ocr_db"
pool_size = 10
max_overflow = 10
```

## Notes
- Images are stored on disk under `uploads/` and paths are stored in the DB.
- PaddleOCR and OpenCV are pinned for Python 3.12 compatibility.
- For production, map persistent volumes and harden configuration and logging.

## License
MIT


