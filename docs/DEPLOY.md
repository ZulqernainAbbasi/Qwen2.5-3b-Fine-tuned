# Deployment Guide

Architecture: **Streamlit Community Cloud** (UI, permanent URL) → **Cloudflare quick tunnel** → **your laptop** (FastAPI + LangGraph agent + ChromaDB + Ollama `car-assistant`).

## One-time setup
1. Install cloudflared: `winget install --id Cloudflare.cloudflared`
2. Build the RAG index: `.\.venv\Scripts\python -m rag.ingest`
3. Make sure `ollama list` shows `car-assistant` (see `finetune/README.md`).
4. Push this repo to GitHub (public). `.env`, `data/chroma`, `outputs/` and bookings are git-ignored.
5. On https://share.streamlit.io, choose **New app**, pick the repo, branch `main`, main file `frontend/app.py`. Under Advanced settings, set the requirements file to `frontend/requirements.txt` (or keep the root one; only streamlit + requests are needed).
6. In the app's **Secrets**, add:
   ```toml
   BACKEND_URL = "https://<printed-url>.trycloudflare.com"
   CAR_API_KEY = "<value from .env>"
   ADMIN_PASSWORD = "<choose one>"
   ```

## Demo-day checklist
1. Open the Ollama app (it must be running).
2. PowerShell in the project folder: `powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1`
3. Copy the printed `PUBLIC BACKEND URL` into Streamlit **Secrets** as `BACKEND_URL` (the quick-tunnel URL changes on every start), save, and reboot the app.
4. Open the Streamlit URL. The red "Backend is offline" banner should be gone.
5. Keep the laptop awake and plugged in while it's in use.

## Troubleshooting
- Offline banner: check that `outputs\backend_url.txt` matches the secret, and that `Invoke-RestMethod <url>/health` returns ok.
- 401 errors: the `CAR_API_KEY` in Secrets must match `.env`.
- Slow first answer: Ollama is loading the model into GPU memory (about 5–10 s).
