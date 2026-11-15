# Pakistan Car Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A RAG + fine-tuned + agentic assistant for new cars in Pakistan. The backend runs on the owner's laptop and the chat UI runs on Streamlit Community Cloud.

**Architecture:** Car data (CSV + markdown docs) is embedded into ChromaDB. A Qwen2.5-3B model is fine-tuned with QLoRA (Unsloth, WSL), exported to GGUF and served by Ollama. A LangGraph agent with six tools runs behind FastAPI, which a Cloudflare quick tunnel exposes to a Streamlit Cloud frontend.

**Tech Stack:** Python 3.11, pandas, chromadb, sentence-transformers (BAAI/bge-small-en-v1.5), rapidfuzz, langgraph, langchain-ollama, fastapi, uvicorn, sqlite3, streamlit, requests, pytest; WSL: unsloth, trl, datasets; Ollama; cloudflared.

**Spec:** `docs/superpowers/specs/2026-11-15-car-assistant-design.md`

## Global Constraints

- Project root: `C:\Users\Zulqernain Abbasi\Documents\CCP-Car-Assistant` (Windows venv `.venv`; WSL venv `~/ccp-ft-venv` for fine-tuning only).
- Brands: Toyota, Honda, Suzuki, KIA, Hyundai, Changan, MG; new cars only.
- Prices in whole PKR (int). Every CSV row has `source_url` and `as_of_date = 2026-11`.
- UI shows "Prices as of Nov 2026, verify with dealer."
- Base model `Qwen/Qwen2.5-3B-Instruct`; Ollama model name `car-assistant`; GGUF quant `Q4_K_M`.
- Embeddings `BAAI/bge-small-en-v1.5`; Chroma path `data/chroma`; collection `cars`.
- No paid services, no hosted LLM APIs.
- Never invent prices or specs. If retrieval returns nothing, reply with the exact refusal `I don't have verified information on that.`
- Agent max 6 tool steps per turn. Bookings need explicit user confirmation.
- Backend auth: header `X-API-Key` must equal env `CAR_API_KEY`.
- PyTorch for the RTX 5060 (sm_120) must be a CUDA 12.8+ build.

## Review Focus

1. **Unknown car/brand** (e.g. "Tesla Model 3 price"): refusal string, no invented number. Tested in Task 6.
2. **Pakistani price units** ("under 50 lakh", "1.2 crore", "45 lac", "6 million"): parsed to PKR correctly. Tested in Task 4.
3. **Misspelled or Roman-Urdu model names** ("corola", "civic rs", "alto ki qeemat"): fuzzy-matched to the right model. Tested in Task 4.
4. **Backend offline / stale tunnel URL:** Streamlit shows an offline banner and doesn't crash. Tested in Task 8.
5. **Bad booking input** (invalid PK phone, past date): tool rejects with a clear message and saves nothing. Tested in Task 4.

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `config.py`, `tests/test_config.py`, `README.md`

**Interfaces:**
- Produces: `config.py` constants `ROOT: Path`, `DATA_DIR`, `CARS_CSV = DATA_DIR/"cars.csv"`, `DOCS_DIR = DATA_DIR/"docs"`, `CHROMA_DIR = DATA_DIR/"chroma"`, `DB_PATH = DATA_DIR/"bookings.db"`, `EMBED_MODEL = "BAAI/bge-small-en-v1.5"`, `LLM_MODEL = os.getenv("LLM_MODEL","car-assistant")`, `REFUSAL = "I don't have verified information on that."`

- [ ] **Step 1:** `git init`, create `.venv` with Python 3.11, write `requirements.txt` (unpinned latest of the Windows stack above) and `.gitignore` (`.venv/ data/chroma/ data/bookings.db *.gguf outputs/ .env`).
- [ ] **Step 2:** Write `tests/test_config.py::test_paths_under_root` asserting every path constant is under `ROOT` and `REFUSAL` equals the exact string.
- [ ] **Step 3:** Implement `config.py`. Run `pytest -q` and expect PASS.
- [ ] **Step 4:** Commit `chore: scaffold project`.

### Task 2: Collect and validate car data

**Files:**
- Create: `data/cars.csv`, `data/docs/*.md`, `data/raw/` (source snapshots), `data/validate.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `cars.csv` columns exactly `brand,model,variant,price_pkr,engine_cc,fuel,transmission,mileage_kmpl,body_type,seats,key_features,source_url,as_of_date`. `fuel ∈ {petrol,hybrid,phev,electric,diesel}`. `body_type ∈ {hatchback,sedan,suv,crossover,pickup,mpv}`. `key_features` is `;`-separated.
- Produces: `data/validate.py::load_cars() -> pandas.DataFrame` (validated, raises `ValueError` listing bad rows).
- Docs: one `<brand>-<model>.md` per model, plus `warranty-<brand>.md`, `maintenance-general.md`, `financing-pakistan.md`, `faq.md`. Each starts with `# Title` and `Source: <url>`.

- [ ] **Step 1:** Write tests: `test_all_seven_brands_present`, `test_price_is_positive_int_between_1m_and_100m`, `test_enums_valid`, `test_every_row_has_source_url`, `test_no_duplicate_brand_model_variant`, `test_each_model_has_doc`.
- [ ] **Step 2:** Run them. Expected FAIL (no data).
- [ ] **Step 3:** Research current Pakistani new-car line-ups (official brand PK sites, PakWheels new-car pages) and fill the CSV, aiming for ≥ 35 models / ≥ 80 variants. Save page snapshots to `data/raw/`. Write the docs. Use blank cells, never guesses, for unknown specs.
- [ ] **Step 4:** Implement `load_cars()`. Run tests and expect PASS.
- [ ] **Step 5:** Commit `data: add Pakistan new-car dataset (Nov 2026)`.

### Task 3: RAG ingest and retriever

**Files:**
- Create: `rag/ingest.py`, `rag/retriever.py`, `tests/test_rag.py`

**Interfaces:**
- Consumes: `load_cars()`, `config.*`
- Produces: `rag/ingest.py::build_index() -> int` (chunks indexed; rebuilds collection). Spec cards are one per CSV row and render all fields as `Brand Model Variant: price PKR …`. Docs are chunked at 500 tokens / 50 overlap. Metadata: `brand, model, price_pkr (rows only), source, kind ∈ {spec,doc}`.
- Produces: `rag/retriever.py::retrieve(query: str, k: int = 5, brand: str|None = None, max_price: int|None = None) -> list[dict]`, where each dict is `{text, source, brand, model, score}`. Drops results with score below `MIN_SCORE = 0.35` (cosine similarity).

- [ ] **Step 1:** Write tests on a 3-row fixture CSV + 1 doc in `tmp_path`: `test_build_index_counts`, `test_retrieve_finds_corolla_for_corolla_price`, `test_brand_filter`, `test_max_price_filter_excludes_expensive`, `test_irrelevant_query_returns_empty` (query "recipe for biryani" → `[]`).
- [ ] **Step 2:** Run tests. Expected FAIL.
- [ ] **Step 3:** Implement with a persistent Chroma client and a SentenceTransformer embedding function (normalize embeddings).
- [ ] **Step 4:** Run tests and expect PASS. Then run `python -m rag.ingest` on the real data and expect it to print the count.
- [ ] **Step 5:** Commit `feat: RAG ingest and retriever`.

### Task 4: Agent tools

**Files:**
- Create: `agent/parsing.py`, `agent/tools.py`, `backend/db.py`, `tests/test_parsing.py`, `tests/test_tools.py`, `tests/test_db.py`

**Interfaces:**
- Produces `agent/parsing.py`:
  - `parse_price(text: str) -> int|None`: lakh/lac = 1e5, crore = 1e7, million/m = 1e6, plain digits with commas.
  - `match_model(name: str) -> tuple[str,str]|None`: (brand, model) via rapidfuzz `WRatio ≥ 80` over `"brand model"` and `"model"` strings from `cars.csv`.
- Produces `backend/db.py`: `init_db(path=DB_PATH)`, `add_booking(kind: Literal["test_drive","service"], **fields) -> int`, `list_bookings() -> list[dict]`, `add_ticket(summary: str) -> int`, `list_tickets() -> list[dict]`.
- Produces `agent/tools.py` (LangChain `@tool` functions, all returning `str`):
  - `search_cars(query: str, brand: str|None=None, max_price: str|None=None, body_type: str|None=None)`: `max_price` goes through `parse_price`. Combines CSV filtering and `retrieve()`. Lists up to 5 variants with price and source, or returns `REFUSAL`.
  - `compare_models(model_a: str, model_b: str)`: markdown table of the cheapest and top variant of each, from the CSV. Uses `REFUSAL` if either doesn't match.
  - `emi_calculator(price: str, down_payment_pct: float = 30, years: int = 5, annual_rate_pct: float = 20)`: standard amortization. Returns down payment, loan, monthly instalment, total interest (PKR, rounded).
  - `book_test_drive(name, phone, model, city, date)` and `book_service(name, phone, model, km: int, date)`: validate phone `^(\+92|0)3\d{9}$` (spaces/dashes stripped), date ISO and ≥ today, model through `match_model`. Return `Booked #<id> …` or `Cannot book: <reason>`.
  - `escalate_to_human(summary: str)` → `Ticket #<id> created; a staff member will contact you.`
  - `ALL_TOOLS: list` with the six tools.

- [ ] **Step 1:** Write parsing tests: `parse_price("under 50 lakh")==5_000_000`, `("1.2 crore")==12_000_000`, `("45 lac")==4_500_000`, `("6 million")==6_000_000`, `("Rs 7,499,000")==7_499_000`, `("cheap")is None`. Model matching: `"corola"→("Toyota","Corolla")`, `"civic rs"→("Honda","Civic")`, `"alto ki qeemat"→("Suzuki","Alto")`, `"tesla model 3"→None`.
- [ ] **Step 2:** Write db and tool tests (monkeypatch `DB_PATH` to `tmp_path`): `test_emi_known_value` (price 5,000,000, 30%, 5y, 20% → monthly 92,727 ± 1), `test_booking_rejects_bad_phone`, `test_booking_rejects_past_date`, `test_booking_saves_valid` (then `list_bookings()` has 1), `test_compare_unknown_returns_refusal`, `test_search_tesla_returns_refusal`, `test_ticket_created`.
- [ ] **Step 3:** Run tests. Expected FAIL.
- [ ] **Step 4:** Implement the three modules.
- [ ] **Step 5:** Run `pytest -q` and expect all PASS.
- [ ] **Step 6:** Commit `feat: agent tools, parsing, bookings db`.

### Task 5: Fine-tuning dataset

**Files:**
- Create: `finetune/build_dataset.py`, `finetune/handwritten.jsonl`, `data/finetune/train.jsonl`, `data/finetune/eval.jsonl`, `tests/test_finetune_data.py`

**Interfaces:**
- Consumes: `load_cars()`, `REFUSAL`
- Produces: JSONL in chat format `{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}`. System prompt constant `SYSTEM_PROMPT` lives in `finetune/build_dataset.py` and is reused by the agent (Task 6).
- Example types and minimum counts: spec Q&A 300, recommendation 200, comparison 150, refusal 120 (unknown cars/specs → `REFUSAL` + offer of what *is* known), Roman Urdu 100, tool-call style examples 100 (assistant emits a Qwen-format tool call JSON for the six tool names). Handwritten ≥ 40.
- `build(seed=42) -> tuple[int,int]`: writes train/eval with a 90/10 split, stratified by type. Eval questions never repeat a train question verbatim.

- [ ] **Step 1:** Write tests: `test_counts_meet_minimums`, `test_every_price_in_answers_exists_in_csv` (regex all `PKR`/`Rs` numbers in assistant turns, each ∈ CSV prices or derived EMI values tagged `#calc`), `test_no_eval_train_overlap`, `test_refusals_use_exact_string`.
- [ ] **Step 2:** Run tests. Expected FAIL.
- [ ] **Step 3:** Implement template generators and write the handwritten file.
- [ ] **Step 4:** Run `python -m finetune.build_dataset` and the tests. Expect PASS.
- [ ] **Step 5:** Commit `feat: fine-tuning dataset builder`.

### Task 6: Fine-tune, export, serve (WSL + Ollama)

**Files:**
- Create: `finetune/setup_wsl.sh`, `finetune/train.py`, `finetune/export_gguf.sh`, `Modelfile`, `finetune/README.md`, `outputs/` (gitignored)

**Interfaces:**
- Consumes: `data/finetune/train.jsonl`
- Produces: Ollama model `car-assistant` (GGUF Q4_K_M, Qwen chat template, `num_ctx 4096`, `temperature 0.3`, `SYSTEM` = `SYSTEM_PROMPT`).
- `train.py` hyperparams: 4-bit load, LoRA r=16, alpha=16, dropout=0, targets all attn+MLP projections, max_seq_len 2048, batch 2 × grad-accum 4, lr 2e-4, 3 epochs, `train_on_responses_only`. Saves `outputs/lora/` and logs loss to `outputs/train_log.csv`.

- [ ] **Step 1:** `setup_wsl.sh`: creates `~/ccp-ft-venv`, installs a CUDA 12.8+ PyTorch build and unsloth. Verify in WSL with `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"` → expected `NVIDIA GeForce RTX 5060 Laptop GPU (12, 0)`.
- [ ] **Step 2:** Smoke run `python finetune/train.py --max_steps 10`. Expected: loss printed, no OOM, peak VRAM < 7.5 GB.
- [ ] **Step 3:** Full run. Expected: final train loss below the step-10 loss, `outputs/lora/` exists.
- [ ] **Step 4:** `export_gguf.sh` merges and saves GGUF Q4_K_M to `outputs/car-assistant-Q4_K_M.gguf`. On Windows, run `ollama create car-assistant -f Modelfile`.
- [ ] **Step 5:** Verify `ollama run car-assistant "Toyota Yaris ki price kya hai?"` gives an answer with a price that exists in the CSV, and `"Tesla Model 3 price in Pakistan"` gives the refusal string.
- [ ] **Step 6:** Commit `feat: fine-tune pipeline and Modelfile` (scripts only, no weights).

### Task 7: LangGraph agent

**Files:**
- Create: `agent/graph.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: `ALL_TOOLS`, `SYSTEM_PROMPT`, `LLM_MODEL`
- Produces: `agent/graph.py::run_agent(session_id: str, message: str) -> dict` returning `{"reply": str, "steps": list[{"tool": str, "args": dict, "result": str}]}`. Memory is per `session_id` (LangGraph `MemorySaver`, thread_id = session_id). Recursion limit enforces ≤ 6 tool steps, and when exceeded the reply is `escalate_to_human` output. `get_llm()` returns `ChatOllama(model=LLM_MODEL, temperature=0.3)` and is monkeypatchable.
- System prompt addition: "Use tools for any price/spec. Before booking, restate details and wait for 'yes'."

- [ ] **Step 1:** Write tests with a fake tool-calling chat model (LangChain `GenericFakeChatModel` or a stub returning scripted `AIMessage(tool_calls=…)`): `test_tool_called_and_trace_recorded`, `test_memory_keeps_session_context`, `test_step_limit_escalates`.
- [ ] **Step 2:** Run tests. Expected FAIL.
- [ ] **Step 3:** Implement with `langgraph.prebuilt.create_react_agent`.
- [ ] **Step 4:** Run tests and expect PASS. Manual check with real Ollama: "Family of 5, budget 80 lakh, compare two options and EMI with 30% down" produces a trace showing `search_cars → compare_models → emi_calculator`.
- [ ] **Step 5:** Commit `feat: LangGraph agent`.

### Task 8: FastAPI backend and Streamlit frontend

**Files:**
- Create: `backend/api.py`, `frontend/app.py`, `frontend/client.py`, `frontend/requirements.txt` (streamlit, requests only), `tests/test_api.py`, `tests/test_client.py`

**Interfaces:**
- Consumes: `run_agent`, `list_bookings`, `list_tickets`, `init_db`
- Produces API: `GET /health` → `{"status":"ok","model":LLM_MODEL}` (no auth). `POST /chat` body `{"session_id": str, "message": str (1–1000 chars)}` → `run_agent` result. `GET /bookings` → `{"bookings": [...], "tickets": [...]}`. Missing or wrong `X-API-Key` → 401.
- Produces `frontend/client.py::BackendClient(base_url: str, api_key: str, timeout: float = 60)` with `.health() -> bool` (False on any exception), `.chat(session_id, message) -> dict`, `.bookings() -> dict`. Raises `BackendOffline` on connection error/timeout.
- `frontend/app.py`: reads `st.secrets["BACKEND_URL"]`, `["CAR_API_KEY"]`, `["ADMIN_PASSWORD"]`. Tabs: Chat (history in `st.session_state`, expandable "Agent steps"), Compare (two selectboxes → sends "compare X and Y"), Admin (password → bookings/tickets tables). Shows the price disclaimer. If offline, shows `st.error("Backend is offline — the demo laptop isn't running. Please try again later.")` and disables input.

- [ ] **Step 1:** Write API tests with `TestClient` and `run_agent` monkeypatched: `test_health_no_auth`, `test_chat_requires_key`, `test_chat_returns_reply_and_steps`, `test_empty_message_422`, `test_bookings_lists`.
- [ ] **Step 2:** Write client tests: `test_health_false_when_unreachable` (base_url `http://127.0.0.1:9`), `test_chat_raises_backend_offline` on the same URL.
- [ ] **Step 3:** Run tests. Expected FAIL.
- [ ] **Step 4:** Implement the API, client and app.
- [ ] **Step 5:** Run tests and expect PASS. Manual check: run `uvicorn backend.api:app` and `streamlit run frontend/app.py` with local secrets; chat works. Stop uvicorn and the offline banner appears.
- [ ] **Step 6:** Commit `feat: FastAPI backend and Streamlit frontend`.

### Task 9: Tunnel, start script, cloud deploy

**Files:**
- Create: `scripts/start_backend.ps1`, `docs/DEPLOY.md`

**Interfaces:**
- `start_backend.ps1`: checks that `ollama list` contains `car-assistant`, starts uvicorn on :8000 (env `CAR_API_KEY` from `.env`), starts `cloudflared tunnel --url http://localhost:8000`, parses and prints the `https://*.trycloudflare.com` URL, and polls `/health` through the tunnel until ok.

- [ ] **Step 1:** Install cloudflared (`winget install Cloudflare.cloudflared`). Write the script.
- [ ] **Step 2:** Run it. Expected: printed URL, and `curl <url>/health` returns `status ok`.
- [ ] **Step 3:** Push the repo to GitHub (public; no data/chroma, no .env). Deploy `frontend/app.py` on Streamlit Community Cloud with secrets. Expected: the public app answers a question through the tunnel.
- [ ] **Step 4:** Write `DEPLOY.md` (demo-day checklist: start script → paste URL into secrets → reboot app).
- [ ] **Step 5:** Commit `feat: deployment scripts and guide`.

### Task 10: Evaluation

**Files:**
- Create: `eval/run_eval.py`, `eval/score.py`, `tests/test_score.py`, `eval/results.csv`, `eval/charts/*.png`

**Interfaces:**
- Consumes: `data/finetune/eval.jsonl`, `retrieve`, `ALL_TOOLS`, Ollama models `qwen2.5:3b-instruct` (base) and `car-assistant`
- Produces `eval/score.py::score(answer: str, expected: str, kind: str) -> dict` with `{"price_correct": bool|None, "refusal_correct": bool|None, "keyword_recall": float}`. Price is correct if every PKR number in the expected answer appears in the answer (±0.5%).
- `run_eval.py --config {base,base_rag,ft_rag}` writes rows `config,id,kind,latency_s,price_correct,refusal_correct,keyword_recall`. A summary prints the mean per config, and the script writes a bar chart.

- [ ] **Step 1:** Write `test_score.py`: exact price → True, wrong price → False, refusal expected + refusal given → True, refusal expected + invented price → False.
- [ ] **Step 2:** Run tests. Expected FAIL. Implement `score.py`. Run again and expect PASS.
- [ ] **Step 3:** `ollama pull qwen2.5:3b-instruct`, then run all three configs. Expected: `results.csv` has 3 × N rows and the charts exist.
- [ ] **Step 4:** Write `docs/model-selection.md` comparing Qwen2.5-3B, Llama-3.2-3B and Phi-3.5-mini (params, license, VRAM for QLoRA, tool-calling, Urdu support) for the report (spec §6.1).
- [ ] **Step 5:** Commit `feat: evaluation, results and model-selection notes`.
