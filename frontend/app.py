"""PakCar Assistant: Streamlit Community Cloud frontend.

Secrets (Streamlit Cloud > App settings > Secrets, or .streamlit/secrets.toml locally):
    BACKEND_URL = "https://xxxx.trycloudflare.com"
    CAR_API_KEY = "..."
    ADMIN_PASSWORD = "..."
"""
import csv
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import BackendClient, BackendOffline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Prices are ex-factory as of Nov 2026, verify with the dealer."
OFFLINE = "Backend is offline — the demo laptop isn't running. Please try again later."


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


@st.cache_data
def model_names() -> list[str]:
    with open(ROOT / "data" / "cars.csv", encoding="utf-8") as f:
        return sorted({f"{r['brand']} {r['model']}" for r in csv.DictReader(f)})


st.set_page_config(page_title="PakCar Assistant", page_icon="🚗", layout="centered")
client = BackendClient(secret("BACKEND_URL"), secret("CAR_API_KEY"), timeout=120)
online = client.health()

st.title("🚗 PakCar Assistant")
st.caption("RAG + fine-tuned Qwen2.5-3B + LangGraph agent · new cars in Pakistan · " + DISCLAIMER)
if not online:
    st.error(OFFLINE)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []


def ask(message: str) -> None:
    st.session_state.history.append({"role": "user", "content": message})
    try:
        with st.spinner("Thinking..."):
            out = client.chat(st.session_state.session_id, message)
        st.session_state.history.append({"role": "assistant", "content": out["reply"], "steps": out.get("steps", [])})
    except BackendOffline:
        st.session_state.history.append({"role": "assistant", "content": OFFLINE, "steps": []})
    except Exception as e:
        st.session_state.history.append({"role": "assistant", "content": f"Something went wrong: {e}", "steps": []})


chat_tab, compare_tab, admin_tab = st.tabs(["💬 Chat", "⚖️ Compare", "🔒 Admin"])

with compare_tab:
    names = model_names()
    c1, c2 = st.columns(2)
    a = c1.selectbox("First car", names, index=names.index("Toyota Corolla") if "Toyota Corolla" in names else 0)
    b = c2.selectbox("Second car", names, index=names.index("Honda Civic") if "Honda Civic" in names else 1)
    if st.button("Compare", disabled=not online):
        ask(f"Compare {a} and {b}")
        st.info("Comparison added to the Chat tab.")

with admin_tab:
    pw = st.text_input("Admin password", type="password")
    if pw and pw == secret("ADMIN_PASSWORD"):
        try:
            data = client.bookings()
            st.subheader("Bookings")
            st.dataframe(data["bookings"], use_container_width=True)
            st.subheader("Escalation tickets")
            st.dataframe(data["tickets"], use_container_width=True)
        except BackendOffline:
            st.error(OFFLINE)
    elif pw:
        st.warning("Wrong password.")

with chat_tab:
    if not st.session_state.history:
        st.markdown("Try: *Best family car under 60 lakh?* · *Corolla vs Civic* · "
                    "*80 lakh car ki 5 saal ki qist?* · *Book a test drive*")
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander(f"Agent steps ({len(msg['steps'])})"):
                    for i, s in enumerate(msg["steps"], 1):
                        st.markdown(f"**{i}. `{s['tool']}`** `{s['args']}`")
                        st.code(s["result"][:1500])
    if prompt := st.chat_input("Ask about any new car in Pakistan...", disabled=not online):
        ask(prompt)
        st.rerun()
