"""PakCar Assistant: Streamlit Community Cloud frontend.

Secrets (Streamlit Cloud > App settings > Secrets, or .streamlit/secrets.toml locally):
    BACKEND_URL = "https://xxxx.trycloudflare.com"
    CAR_API_KEY = "..."
    ADMIN_PASSWORD = "..."
"""
import html
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import fmt_pkr, load_models, models_in_text  # noqa: E402
from client import BackendClient, BackendOffline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Ex-factory prices, Nov 2026. Please confirm with an authorised dealer."
OFFLINE = "The assistant is offline right now (the demo laptop isn't running). You can still browse the Showroom."
BRAND_COLORS = {"Toyota": "#eb0a1e", "Honda": "#cc0000", "Suzuki": "#1d4f9c", "KIA": "#05141f",
                "Hyundai": "#002c5f", "Changan": "#0a5cad", "MG": "#9e1b32"}
EXAMPLES = ["Best family car under 60 lakh?", "Toyota Yaris price", "Corolla vs Civic",
            "Cultus oil filter price", "80 lakh car ki 5 saal ki qist?", "Book a test drive"]

st.set_page_config(page_title="PakCar Assistant", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.hero { background: radial-gradient(1200px 300px at 10% -40%, rgba(212,175,55,.25), transparent),
        linear-gradient(135deg, #111826 0%, #0b1020 100%); border: 1px solid rgba(255,255,255,.08);
        border-radius: 22px; padding: 26px 30px; margin-bottom: 18px; }
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -.5px; color: #f5f7fb; }
.hero h1 span { color: #d4af37; }
.hero p { color: #9aa4b8; margin: 6px 0 14px; font-size: 1rem; }
.chip { display: inline-block; padding: 4px 11px; margin: 0 6px 6px 0; border-radius: 999px; font-size: .78rem;
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); color: #d5dbe7; }
.chip.gold { background: rgba(212,175,55,.14); border-color: rgba(212,175,55,.35); color: #e9cf7a; }
.card { background: #121a2a; border: 1px solid rgba(255,255,255,.08); border-radius: 18px; overflow: hidden;
        margin-bottom: 14px; transition: transform .15s ease, border-color .15s ease; }
.card:hover { transform: translateY(-2px); border-color: rgba(212,175,55,.45); }
.card .img { height: 170px; background-size: cover; background-position: center; background-color: #0e1422; }
.card .ph { height: 170px; display: flex; align-items: center; justify-content: center; font-size: 2.2rem;
            font-weight: 800; color: rgba(255,255,255,.85); }
.card .body { padding: 12px 15px 14px; }
.card .brand { font-size: .72rem; text-transform: uppercase; letter-spacing: 1.4px; color: #8b95a9; }
.card .name { font-size: 1.12rem; font-weight: 700; color: #f2f4f8; margin: 2px 0 4px; }
.card .price { color: #d4af37; font-weight: 700; font-size: .98rem; margin-bottom: 8px; }
.credit { font-size: .64rem; color: #6c768a; padding: 0 15px 10px; }
.credit a { color: #7f8aa0; }
.mini { display: flex; gap: 10px; overflow-x: auto; padding: 6px 0 2px; }
.mini .m { min-width: 170px; max-width: 170px; background: #121a2a; border: 1px solid rgba(255,255,255,.08);
           border-radius: 14px; overflow: hidden; }
.mini .m .img { height: 92px; background-size: cover; background-position: center; }
.mini .m .t { padding: 7px 10px 9px; font-size: .8rem; color: #e6e9f0; font-weight: 600; }
.mini .m .t span { display: block; color: #d4af37; font-weight: 600; font-size: .76rem; margin-top: 2px; }
.status { padding: 10px 12px; border-radius: 12px; font-size: .85rem; margin-bottom: 10px; }
.status.on { background: rgba(46,204,113,.10); border: 1px solid rgba(46,204,113,.35); color: #8ee0b0; }
.status.off { background: rgba(231,76,60,.10); border: 1px solid rgba(231,76,60,.35); color: #f0a198; }
.metric { background: #121a2a; border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 16px 18px; }
.metric .v { font-size: 1.8rem; font-weight: 800; color: #f2f4f8; } .metric .l { color: #8b95a9; font-size: .8rem; }
.spec td, .spec th { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,.07); font-size: .9rem; }
.spec { width: 100%; border-collapse: collapse; } .spec th { color: #8b95a9; text-align: left; font-weight: 500; }
[data-testid="stChatMessage"] { background: #101827; border: 1px solid rgba(255,255,255,.06); border-radius: 16px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #121a2a; border-radius: 12px 12px 0 0; padding: 8px 18px; }
</style>
""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


@st.cache_data
def catalog() -> list[dict]:
    return load_models(ROOT)


@st.cache_data(ttl=20)
def backend_online(url: str, key: str) -> bool:
    return BackendClient(url, key).health()


def esc(x) -> str:
    return html.escape(str(x))


def image_div(m: dict, cls: str = "img") -> str:
    if m.get("image"):
        return f'<div class="{cls}" style="background-image:url(\'{esc(m["image"]["image_url"])}\')"></div>'
    color = BRAND_COLORS.get(m["brand"], "#2b3550")
    initials = "".join(w[0] for w in m["model"].split()[:2]).upper()
    return f'<div class="ph" style="background:linear-gradient(135deg,{color},#0b1020)">{esc(initials)}</div>'


def credit(m: dict) -> str:
    img = m.get("image")
    if not img:
        return ""
    return (f'<div class="credit">Photo: {esc(img["artist"][:60])} · <a href="{esc(img["file_page"])}" '
            f'target="_blank">{esc(img["license"])}</a>, Wikimedia Commons (model shown may differ in year/market)</div>')


def price_range(m: dict) -> str:
    if m["min_price"] == m["max_price"]:
        return f"Rs {fmt_pkr(m['min_price'])}"
    return f"Rs {fmt_pkr(m['min_price'])} – {fmt_pkr(m['max_price'])}"


def car_card(m: dict) -> str:
    chips = "".join(f'<span class="chip">{esc(x)}</span>' for x in
                    sorted(m["fuels"]) + sorted(m["body_types"]) + [f"{max(m['seats'])} seats"])
    return (f'<div class="card">{image_div(m)}<div class="body"><div class="brand">{esc(m["brand"])}</div>'
            f'<div class="name">{esc(m["model"])}</div><div class="price">{price_range(m)}</div>{chips}'
            f'</div>{credit(m)}</div>')


def mini_cards(models: list[dict]) -> str:
    cards = "".join(f'<div class="m">{image_div(m)}<div class="t">{esc(m["brand"])} {esc(m["model"])}'
                    f'<span>{price_range(m)}</span></div></div>' for m in models[:5])
    return f'<div class="mini">{cards}</div>'


# ---------------------------------------------------------------- state & backend
client = BackendClient(secret("BACKEND_URL"), secret("CAR_API_KEY"), timeout=120)
online = backend_online(secret("BACKEND_URL"), secret("CAR_API_KEY"))
MODELS = catalog()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []


def ask(message: str) -> None:
    st.session_state.history.append({"role": "user", "content": message})
    try:
        with st.spinner("Checking prices, specs and tools..."):
            out = client.chat(st.session_state.session_id, message)
        st.session_state.history.append({"role": "assistant", "content": out["reply"], "steps": out.get("steps", [])})
    except BackendOffline:
        st.session_state.history.append({"role": "assistant", "content": OFFLINE, "steps": []})
    except Exception as e:
        st.session_state.history.append({"role": "assistant", "content": f"Something went wrong: {e}", "steps": []})


if st.session_state.get("pending"):
    ask(st.session_state.pop("pending"))
    st.toast("Answer added in the Assistant tab", icon="💬")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🚗 PakCar Assistant")
    if online:
        st.markdown('<div class="status on">● Assistant online · fine-tuned Qwen2.5-3B</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status off">● Assistant offline</div>', unsafe_allow_html=True)
    if st.button("✨ New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.markdown("**Try asking**")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex-{ex}", use_container_width=True, disabled=not online):
            st.session_state.pending = ex
            st.rerun()
    st.markdown("---")
    st.caption("How it works: a fine-tuned Qwen2.5-3B model answers using RAG over sourced Pakistani car data, "
               "and a LangGraph agent uses tools to search, compare, calculate EMIs, look up parts and book test drives.")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------- hero
n_brands = len({m["brand"] for m in MODELS})
n_variants = sum(len(m["variants"]) for m in MODELS)
st.markdown(f"""
<div class="hero"><h1>Find your next car in <span>Pakistan</span></h1>
<p>Ask anything about new cars: prices, comparisons, monthly instalments, spare parts and test drives. English or Roman Urdu.</p>
<span class="chip gold">{len(MODELS)} models</span><span class="chip gold">{n_variants} variants</span>
<span class="chip gold">{n_brands} brands</span><span class="chip">Prices as of Nov 2026</span>
<span class="chip">RAG + fine-tuned LLM + AI agent</span></div>
""", unsafe_allow_html=True)

chat_tab, showroom_tab, compare_tab, admin_tab = st.tabs(["💬 Assistant", "🚘 Showroom", "⚖️ Compare", "🛡️ Admin"])

# ---------------------------------------------------------------- assistant
with chat_tab:
    if not online:
        st.warning(OFFLINE)
    if not st.session_state.history:
        st.markdown("<p style='color:#8b95a9'>Assalam-o-Alaikum! Ask about any Toyota, Honda, Suzuki, KIA, Hyundai, "
                    "Changan or MG car, or pick an example from the sidebar.</p>", unsafe_allow_html=True)
    for msg in st.session_state.history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🚗"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                mentioned = models_in_text(msg["content"], MODELS)
                if mentioned:
                    st.markdown(mini_cards(mentioned), unsafe_allow_html=True)
                if msg.get("steps"):
                    tools_used = " → ".join(f"`{s['tool']}`" for s in msg["steps"])
                    with st.expander(f"🤖 Agent steps: {tools_used}"):
                        for i, s in enumerate(msg["steps"], 1):
                            st.markdown(f"**{i}. `{s['tool']}`** `{s['args']}`")
                            st.code(s["result"][:1500], language=None)
    if prompt := st.chat_input("Ask about prices, comparisons, EMI, parts or bookings...", disabled=not online):
        ask(prompt)
        st.rerun()

# ---------------------------------------------------------------- showroom
with showroom_tab:
    c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1.5])
    brands = c1.multiselect("Brand", sorted({m["brand"] for m in MODELS}))
    lo, hi = c2.slider("Budget (lakh PKR)", 25, 170, (25, 170), step=5)
    body = c3.selectbox("Body type", ["Any", "hatchback", "sedan", "crossover", "suv", "mpv", "pickup"])
    fuel = c4.selectbox("Fuel", ["Any", "petrol", "hybrid", "electric", "diesel"])
    shown = [m for m in MODELS
             if (not brands or m["brand"] in brands)
             and m["min_price"] <= hi * 100_000 and m["max_price"] >= lo * 100_000
             and (body == "Any" or body in m["body_types"]) and (fuel == "Any" or fuel in m["fuels"])]
    shown.sort(key=lambda m: m["min_price"])
    st.caption(f"{len(shown)} models · {DISCLAIMER}")
    cols = st.columns(3)
    for i, m in enumerate(shown):
        with cols[i % 3]:
            st.markdown(car_card(m), unsafe_allow_html=True)
            with st.expander(f"{len(m['variants'])} variant{'s' if len(m['variants']) != 1 else ''}"):
                rows = "".join(f"<tr><td>{esc(v['variant'])}</td><td style='color:#d4af37'>Rs {v['price']:,}</td>"
                               f"<td>{esc(v['transmission'])}</td></tr>" for v in m["variants"])
                st.markdown(f"<table class='spec'><tr><th>Variant</th><th>Price</th><th>Gearbox</th></tr>{rows}</table>",
                            unsafe_allow_html=True)
                if m["features"]:
                    st.caption("Features: " + ", ".join(sorted(m["features"])))
            if st.button(f"Ask about the {m['model']}", key=f"ask-{m['brand']}-{m['model']}",
                         use_container_width=True, disabled=not online):
                st.session_state.pending = f"Tell me about the {m['brand']} {m['model']}: all variants, prices and who it suits."
                st.rerun()

# ---------------------------------------------------------------- compare
with compare_tab:
    names = [f"{m['brand']} {m['model']}" for m in MODELS]
    by_name = dict(zip(names, MODELS))
    c1, c2 = st.columns(2)
    a = c1.selectbox("First car", names, index=names.index("Toyota Corolla") if "Toyota Corolla" in names else 0)
    b = c2.selectbox("Second car", names, index=names.index("Honda Civic") if "Honda Civic" in names else 1)
    ma, mb = by_name[a], by_name[b]
    c1.markdown(car_card(ma), unsafe_allow_html=True)
    c2.markdown(car_card(mb), unsafe_allow_html=True)

    def spec(m, key):
        if key == "Price range":
            return f"Rs {m['min_price']:,} – {m['max_price']:,}"
        if key == "Variants":
            return len(m["variants"])
        if key == "Engine":
            cc = sorted({v["engine_cc"] for v in m["variants"] if v["engine_cc"]})
            return ", ".join(f"{c} cc" for c in cc) or "Electric"
        if key == "Fuel":
            return ", ".join(sorted(m["fuels"]))
        if key == "Gearbox":
            return ", ".join(sorted({v["transmission"] for v in m["variants"]}))
        if key == "Seats":
            return ", ".join(str(s) for s in sorted(m["seats"]))
        if key == "Fuel average":
            return ", ".join(sorted({v["mileage"] + " km/l" for v in m["variants"] if v["mileage"]})) or "not listed"
        return ""

    keys = ["Price range", "Variants", "Engine", "Fuel", "Gearbox", "Seats", "Fuel average"]
    rows = "".join(f"<tr><th>{k}</th><td>{esc(spec(ma, k))}</td><td>{esc(spec(mb, k))}</td></tr>" for k in keys)
    st.markdown(f"<table class='spec'><tr><th></th><th>{esc(a)}</th><th>{esc(b)}</th></tr>{rows}</table>",
                unsafe_allow_html=True)
    st.write("")
    if st.button("🤖 Ask the assistant which one suits me", disabled=not online):
        st.session_state.pending = f"Compare {a} and {b}. Which is better value for a Pakistani family?"
        st.rerun()

# ---------------------------------------------------------------- admin
with admin_tab:
    st.markdown("#### Dealer staff dashboard")
    st.caption("The agent books test drives and services on its own and hands complaints to a human. "
               "Staff use this page to review everything the agent did (human-in-the-loop).")
    pw = st.text_input("Admin password", type="password")
    if pw and pw == secret("ADMIN_PASSWORD"):
        try:
            data = client.bookings()
            bks, tks = data["bookings"], data["tickets"]
            m1, m2, m3, m4 = st.columns(4)
            for col, label, value in [(m1, "Total bookings", len(bks)),
                                      (m2, "Test drives", sum(b["kind"] == "test_drive" for b in bks)),
                                      (m3, "Service bookings", sum(b["kind"] == "service" for b in bks)),
                                      (m4, "Escalation tickets", len(tks))]:
                col.markdown(f'<div class="metric"><div class="v">{value}</div><div class="l">{label}</div></div>',
                             unsafe_allow_html=True)
            st.write("")
            st.markdown("**Bookings made by the agent**")
            st.dataframe(bks, use_container_width=True, hide_index=True)
            st.markdown("**Conversations handed to staff**")
            st.dataframe(tks, use_container_width=True, hide_index=True)
        except BackendOffline:
            st.error(OFFLINE)
    elif pw:
        st.error("Wrong password.")
