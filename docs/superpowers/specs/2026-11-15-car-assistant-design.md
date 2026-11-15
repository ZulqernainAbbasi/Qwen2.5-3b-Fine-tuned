# PakWheels-style Car Assistant — CCP Project Design

**Date:** 2026-11-15  **Owner:** Zulqernain Abbasi
**Status:** Draft for approval

## 1. Intent

A course (CCP) project showing the whole modern LLM pipeline on one realistic use case:
an AI assistant for **new cars sold in Pakistan**. The required parts are: (1) pick a
pretrained LLM, (2) RAG over a custom dataset, (3) fine-tune it, (4) deploy it, and
(5) add agentic AI so the system carries out workflows with little human help.

Success means evaluators can open a public link, ask real questions about Pakistani
cars, get grounded answers, and watch the agent finish multi-step tasks (recommend,
compare, calculate a loan, book a test drive). The report should show measurable
gains from RAG and from fine-tuning.

## 2. Goals / Non-goals / Anti-goals

**Goals**
- Real brands in the Pakistani market: Toyota, Honda, Suzuki, KIA, Hyundai, Changan, MG
  (current new-car line-ups: variants, prices, engine, mileage, key features).
- Every requirement (1–5) is clearly visible and demoable.
- Runs on the owner's laptop. No paid services.
- A three-way evaluation for the report.

**Non-goals**
- Used-car marketplace and price prediction.
- Live price scraping. The data is a snapshot labelled "prices as of Nov 2026".
- Real email/SMS sending (simulated; the agent writes an "outbox" entry).
- User accounts or authentication beyond a simple admin password.

**Anti-goals (a failure even if it technically works)**
- Inventing prices or specs that aren't in the dataset. When data is missing, the
  assistant must say "I don't have that information" rather than guess.
- Relying on a hosted LLM API (OpenAI, Groq, etc.), which would undermine the
  "our own fine-tuned, deployed model" requirement.

## 3. Constraints

- **Hardware:** Windows 11 laptop, NVIDIA RTX 5060 Laptop GPU (8 GB VRAM, Blackwell
  sm_120 → needs PyTorch built for CUDA 12.8+), 16 GB RAM, ~137 GB free disk.
  WSL2 Ubuntu is installed (used for fine-tuning). Ollama is already installed on Windows.
- **Budget:** zero. Hugging Face Spaces compute is no longer free (checked Nov 2026);
  Streamlit Community Cloud is free but limited to ~2.7 GB RAM and no GPU, so it
  can't run the model itself.
- Python 3.11, git available.

## 4. Approaches considered

| | Approach | Trade-off | Decision |
|---|---|---|---|
| A | Everything local + tunnel | Free and fast; needs laptop on; no permanent UI URL | Rejected by owner (wanted a hosted deployment) |
| B | Host on Hugging Face Spaces (ZeroGPU) | Permanent and fast, but needs PRO ($9/mo) | Rejected: not free |
| C | Hosted API LLM for the agent | Smarter and faster, but undercuts the fine-tuning requirement | Rejected |
| **D** | **Streamlit Cloud UI + laptop backend over Cloudflare tunnel** | Free, permanent UI URL, GPU-speed answers; backend only works while laptop is on | **Chosen** |

Fallback inside D: if the backend is unreachable, the Streamlit app shows a clear
"backend offline" banner instead of crashing. (A tiny 0.5B on-cloud mode was
considered and left out to keep scope tight.)

## 5. Architecture

```
Streamlit Community Cloud (free)              Owner's laptop
┌──────────────────────────────┐   HTTPS   ┌──────────────────────────────────────┐
│ app.py                       │──────────►│ Cloudflare quick tunnel              │
│  • Chat tab                  │  (API key │   └► FastAPI (backend/api.py)        │
│  • Compare tab               │   header) │        ├ LangGraph agent             │
│  • Admin tab (bookings)      │           │        │   tools ─► RAG (ChromaDB)   │
└──────────────────────────────┘           │        │         ─► SQLite (bookings)│
                                           │        │         ─► calculators      │
                                           │        └ Ollama: car-assistant model │
                                           └──────────────────────────────────────┘
```

The backend URL and a shared API key are stored in Streamlit secrets.

## 6. Components

### 6.1 Base model (requirement 1)
- **Qwen2.5-3B-Instruct.** It fits QLoRA training in 8 GB, supports tool calling,
  handles Roman Urdu reasonably. Licence: Qwen Research License, which permits academic/research use like this course project but restricts commercial use (corrected 2026-11-15; Apache 2.0 covers the 0.5B, 1.5B and 7B sizes).
- The report compares it with Llama-3.2-3B and Phi-3.5-mini on size, license,
  tool-calling ability and VRAM use.

### 6.2 Dataset & RAG (requirement 2)
- **Sources:** official brand websites (Pakistan), PakWheels new-car pages, and brand
  warranty/service pages. Collected with the web tools and saved as raw files plus a
  cleaned set.
- **Structured data:** `data/cars.csv` with one row per variant: brand, model, variant,
  price_pkr, engine_cc, fuel, transmission, mileage_kmpl, body_type, seats, key_features,
  source_url, as_of_date.
- **Unstructured docs:** `data/docs/*.md` covering one overview per model, warranty
  policies, maintenance schedules, financing basics and FAQs.
- **Pipeline:** each CSV row becomes a text "spec card", and docs are chunked to about
  500 tokens with 50 overlap. Embeddings use `BAAI/bge-small-en-v1.5` (CPU-friendly)
  and are stored in ChromaDB (persistent, on disk). Retrieval is top-k = 5 with
  metadata filters (brand, price range).
- Every answer cites its source (model/document name).

### 6.3 Fine-tuning (requirement 3)
- **Method:** QLoRA (4-bit) with Unsloth inside WSL2. LoRA r = 16, 2–3 epochs.
- **Training data:** about 800–1,200 instruction pairs in `data/finetune/train.jsonl`,
  generated from the CSV/docs using templates plus hand-written examples. They cover
  advisor-style recommendations, comparisons, "I don't know" refusals for missing
  data, Roman Urdu/English questions, and correct tool-call formatting.
  A held-out set of about 100 questions goes in `eval.jsonl`.
- **Output:** the LoRA adapter is merged and exported to **GGUF (Q4_K_M)**, then
  registered in Ollama as `car-assistant` through a Modelfile.

### 6.4 Agent (requirement 5)
- LangGraph agent loop using `car-assistant` (via Ollama), with these tools:
  1. `search_cars(query, brand?, max_price?, body_type?)`: RAG + CSV filter
  2. `compare_models(model_a, model_b)`: builds a spec table
  3. `emi_calculator(price, down_payment_pct, years, rate)`: loan instalment
  4. `book_test_drive(name, phone, model, city, date)`: writes to SQLite
  5. `book_service(name, phone, model, km, date)`: writes to SQLite
  6. `escalate_to_human(summary)`: logs a ticket for staff
- **Autonomy:** the agent fills in missing details by asking the user, chains tools by
  itself (for example search → compare → EMI → book), and confirms before saving a
  booking.
- Maximum 6 tool steps per turn to prevent loops.

### 6.5 Deployment (requirement 4)
- **Backend:** FastAPI with `POST /chat` (session id + message → reply + tool trace),
  `GET /bookings` (admin), `GET /health`, all protected with an `X-API-Key` header.
- **Tunnel:** `cloudflared tunnel --url http://localhost:8000` (free quick tunnel).
  The URL changes each run, and a script prints it for pasting into Streamlit secrets.
- **Frontend:** a Streamlit app in a public GitHub repo, deployed to Streamlit
  Community Cloud. It has Chat (with an expandable "agent steps" panel), Compare and
  Admin (password) tabs.
- `start_backend.ps1` launches Ollama, FastAPI and the tunnel with one command.

### 6.6 Evaluation (for the report)
- Run the held-out questions through three setups: base model, base + RAG, and
  fine-tuned + RAG.
- **Metrics:** factual accuracy against the CSV (automatic number/field matching),
  refusal correctness, tool-call success rate, and latency.
- **Output:** `eval/results.csv` plus charts.

## 7. Repository layout (in `Documents\CCP-Car-Assistant`)

```
data/raw/ data/cars.csv data/docs/ data/finetune/
rag/ingest.py rag/retriever.py
finetune/build_dataset.py finetune/train.py (WSL) finetune/export_gguf.sh Modelfile
agent/graph.py agent/tools.py
backend/api.py backend/db.py
frontend/app.py frontend/requirements.txt
eval/run_eval.py
scripts/start_backend.ps1
docs/ (this design, the plan, report notes)
```

## 8. Decided by the owner
- Car-company domain, real Pakistani brands.
- Work runs on the owner's laptop. No paid services.
- Deployment: Streamlit Cloud UI + laptop backend (not HF, not local-only).

## 9. Calls made by Claude (open to change)
- Qwen2.5-3B base, Unsloth QLoRA in WSL, Ollama serving, ChromaDB + bge-small,
  LangGraph, FastAPI, SQLite.
- Brand list and the six agent tools above.
- Email/SMS simulated, not sent.
- No on-cloud fallback model.

## 10. Left to the builder
- Exact prompt templates, chunk sizes and hyperparameter tweaks after the first runs.
- UI styling.
