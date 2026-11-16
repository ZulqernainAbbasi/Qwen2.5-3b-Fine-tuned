# Pakistan Car Assistant (CCP Project)

RAG + fine-tuned LLM + agentic AI assistant for new cars sold in Pakistan.

- Design: `docs/superpowers/specs/2026-11-15-car-assistant-design.md`
- Plan: `docs/superpowers/plans/2026-11-15-car-assistant-plan.md`
- Deploy guide: `docs/DEPLOY.md`

Quick start (Windows):
```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest
```

## What's inside
| Requirement | Where |
|---|---|
| 1. Pretrained LLM | Qwen2.5-3B-Instruct (`docs/model-selection.md`) |
| 2. RAG on custom data | `data/cars.csv`, `data/docs/`, `rag/` (ChromaDB + bge-small) |
| 3. Fine-tuning | `finetune/` (QLoRA with Unsloth in WSL; dataset builder + 1,300 examples) |
| 4. Deployment | `backend/api.py` (FastAPI), `frontend/app.py` (Streamlit Cloud), `scripts/start_backend.ps1` (tunnel), `docs/DEPLOY.md` |
| 5. Agentic AI | `agent/graph.py` (LangGraph) + `agent/tools.py` (search, compare, EMI, parts & service, bookings, escalation) |
| Frontend | `frontend/app.py`: Assistant chat with car photos, Showroom gallery, Compare, Admin dashboard |
| Car photos | `data/images.csv` (Wikimedia Commons, free licences, credited in the app; `python -m scripts.fetch_images`) |
| Evaluation | `eval/run_eval.py`, results in `eval/results.csv` and `eval/charts/` |
