"""PakCar Assistant: Streamlit Community Cloud frontend.

Secrets (Streamlit Cloud > App settings > Secrets, or .streamlit/secrets.toml locally):
    BACKEND_URL = "https://xxxx.trycloudflare.com"
    CAR_API_KEY = "..."
    ADMIN_PASSWORD = "..."
"""
import csv
import html
import io
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from catalog import fmt_pkr, load_models, models_in_text  # noqa: E402
from client import BackendClient, BackendOffline  # noqa: E402
from agent.finance import SBP_NOTE, emi_by_years, quote, sbp_terms  # noqa: E402

DISCLAIMER = "Ex-factory prices, Nov 2026. Please confirm with an authorised dealer."
OFFLINE = "The assistant is offline right now (the demo laptop isn't running). Showroom, Financing and Compare still work."
BRAND_COLORS = {"Toyota": "#eb0a1e", "Honda": "#cc0000", "Suzuki": "#1d4f9c", "KIA": "#05141f",
                "Hyundai": "#002c5f", "Changan": "#0a5cad", "MG": "#9e1b32"}
EXAMPLES = ["Best family car under 60 lakh?", "Toyota Yaris price", "Corolla vs Civic",
            "Bank options for Suzuki Alto, 5 years", "Cultus oil filter price", "Book a test drive"]
YEAR_OPTIONS = [1, 2, 3, 4, 5, 6, 7]

st.set_page_config(page_title="PakCar Assistant", page_icon=":material/speed:", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1280px; }
.logo { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.15rem; color:#f2f4f8; margin-bottom:10px; }
.logo .mark { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
              background:linear-gradient(135deg,#d4af37,#8a6d1f); color:#0b1020; font-weight:900; font-size:.9rem; }
.hero { background: radial-gradient(1200px 300px at 10% -40%, rgba(212,175,55,.25), transparent),
        linear-gradient(135deg, #111826 0%, #0b1020 100%); border: 1px solid rgba(255,255,255,.08);
        border-radius: 22px; padding: 24px 30px; margin-bottom: 16px; }
.hero h1 { font-size: 2.05rem; font-weight: 800; margin: 0; letter-spacing: -.5px; color: #f5f7fb !important; }
.hero h1 span { color: #d4af37; }
.hero p { color: #9aa4b8; margin: 6px 0 12px; font-size: 1rem; }
.chip { display: inline-block; padding: 4px 11px; margin: 0 6px 6px 0; border-radius: 999px; font-size: .78rem;
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); color: #d5dbe7; }
.chip.gold { background: rgba(212,175,55,.14); border-color: rgba(212,175,55,.35); color: #e9cf7a; }
.card { background: #121a2a; border: 1px solid rgba(255,255,255,.08); border-radius: 18px; overflow: hidden;
        margin-bottom: 8px; transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease; }
.card:hover { transform: translateY(-3px); border-color: rgba(212,175,55,.5); box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.card.fav { border-color: rgba(212,175,55,.7); }
.card .img { height: 175px; background-size: cover; background-position: center; background-color: #0e1422; }
.card .ph { height: 175px; display: flex; align-items: center; justify-content: center; font-size: 2.2rem;
            font-weight: 800; color: rgba(255,255,255,.85); }
.card .body { padding: 12px 15px 10px; }
.card .brand { font-size: .72rem; text-transform: uppercase; letter-spacing: 1.4px; color: #8b95a9; }
.card .name { font-size: 1.12rem; font-weight: 700; color: #f2f4f8; margin: 2px 0 4px; }
.card .price { color: #d4af37; font-weight: 700; font-size: .98rem; margin-bottom: 8px; }
.credit { font-size: .64rem; color: #6c768a; padding: 0 15px 10px; }
.credit a { color: #7f8aa0; }
.mini { display: flex; gap: 10px; overflow-x: auto; padding: 6px 0 2px; }
.mini .m { min-width: 170px; max-width: 170px; background: #121a2a; border: 1px solid rgba(255,255,255,.08);
           border-radius: 14px; overflow: hidden; }
.mini .m .img, .mini .m .ph { height: 92px; background-size: cover; background-position: center; font-size: 1.4rem; }
.mini .m .t { padding: 7px 10px 9px; font-size: .8rem; color: #e6e9f0; font-weight: 600; }
.mini .m .t span { display: block; color: #d4af37; font-weight: 600; font-size: .76rem; margin-top: 2px; }
.status { padding: 10px 12px; border-radius: 12px; font-size: .85rem; margin-bottom: 10px; }
.status.on { background: rgba(46,204,113,.10); border: 1px solid rgba(46,204,113,.35); color: #8ee0b0; }
.status.off { background: rgba(231,76,60,.10); border: 1px solid rgba(231,76,60,.35); color: #f0a198; }
.metric { background: #121a2a; border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 14px 16px; }
.metric .v { font-size: clamp(1rem, 1.6vw, 1.35rem); font-weight: 800; color: #f2f4f8; white-space: nowrap; } .metric .l { color: #8b95a9; font-size: .78rem; }
.metric.gold .v { color: #d4af37; }
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


def image_div(m: dict) -> str:
    if m.get("image"):
        return f'<div class="img" style="background-image:url(\'{esc(m["image"]["image_url"])}\')"></div>'
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


def car_card(m: dict, fav: bool = False) -> str:
    chips = "".join(f'<span class="chip">{esc(x)}</span>' for x in
                    sorted(m["fuels"]) + sorted(m["body_types"]) + [f"{max(m['seats'])} seats"])
    return (f'<div class="card{" fav" if fav else ""}">{image_div(m)}<div class="body">'
            f'<div class="brand">{esc(m["brand"])}</div><div class="name">{esc(m["model"])}</div>'
            f'<div class="price">{price_range(m)}</div>{chips}</div>{credit(m)}</div>')


def mini_cards(models: list[dict]) -> str:
    cards = "".join(f'<div class="m">{image_div(m)}<div class="t">{esc(m["brand"])} {esc(m["model"])}'
                    f'<span>{price_range(m)}</span></div></div>' for m in models[:5])
    return f'<div class="mini">{cards}</div>'


def metric(col, label: str, value: str, gold: bool = False) -> None:
    col.markdown(f'<div class="metric{" gold" if gold else ""}"><div class="v">{esc(value)}</div>'
                 f'<div class="l">{esc(label)}</div></div>', unsafe_allow_html=True)


def key_of(m: dict) -> str:
    return f"{m['brand']} {m['model']}"


# ---------------------------------------------------------------- state & backend
client = BackendClient(secret("BACKEND_URL"), secret("CAR_API_KEY"), timeout=120)
online = backend_online(secret("BACKEND_URL"), secret("CAR_API_KEY"))
MODELS = catalog()
BY_NAME = {key_of(m): m for m in MODELS}
NAMES = sorted(BY_NAME)
ss = st.session_state
ss.setdefault("session_id", str(uuid.uuid4()))
ss.setdefault("history", [])
ss.setdefault("shortlist", [])
ss.setdefault("fin_car", "Suzuki Alto")
ss.setdefault("cmp_a", "Toyota Corolla")
ss.setdefault("cmp_b", "Honda Civic")


def ask(message: str) -> None:
    ss.history.append({"role": "user", "content": message})
    try:
        with st.spinner("Checking prices, banks, parts and tools..."):
            out = client.chat(ss.session_id, message)
        ss.history.append({"role": "assistant", "content": out["reply"], "steps": out.get("steps", [])})
    except BackendOffline:
        ss.history.append({"role": "assistant", "content": OFFLINE, "steps": []})
    except Exception as e:
        ss.history.append({"role": "assistant", "content": f"Something went wrong: {e}", "steps": []})


def queue(prompt: str, where: str = "Assistant") -> None:
    ss.pending = prompt
    ss.toast_msg = f"Answer added in the {where} tab"


def set_finance(name: str) -> None:
    ss.fin_car = name
    ss.toast_msg = f"{name} loaded in the Financing tab"


def set_compare(a: str, b: str) -> None:
    ss.cmp_a, ss.cmp_b = a, b
    ss.toast_msg = "Loaded in the Compare tab"


def toggle_shortlist(name: str) -> None:
    if name in ss.shortlist:
        ss.shortlist.remove(name)
    else:
        ss.shortlist.append(name)


if ss.get("pending"):
    ask(ss.pop("pending"))
if ss.get("toast_msg"):
    st.toast(ss.pop("toast_msg"), icon=":material/check_circle:")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="logo"><div class="mark">PC</div>PakCar Assistant</div>', unsafe_allow_html=True)
    if online:
        st.markdown('<div class="status on">● Assistant online · fine-tuned Qwen2.5-3B</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status off">● Assistant offline</div>', unsafe_allow_html=True)
    if st.button("New conversation", icon=":material/add_comment:", width="stretch"):
        ss.session_id = str(uuid.uuid4())
        ss.history = []
        st.rerun()
    st.markdown("**Try asking**")
    for ex in EXAMPLES:
        st.button(ex, key=f"ex-{ex}", width="stretch", disabled=not online, on_click=queue, args=(ex,))
    if ss.shortlist:
        st.markdown("---")
        st.markdown(f"**Your shortlist ({len(ss.shortlist)})**")
        for name in ss.shortlist:
            st.caption(f"• {name} — {price_range(BY_NAME[name])}")
        if len(ss.shortlist) >= 2:
            st.button("Compare first two", icon=":material/compare_arrows:", width="stretch",
                      on_click=set_compare, args=(ss.shortlist[0], ss.shortlist[1]))
    st.markdown("---")
    st.caption("How it works: a fine-tuned Qwen2.5-3B model answers using RAG over sourced Pakistani car data, "
               "and a LangGraph agent uses tools to search, compare, quote bank financing, look up parts and "
               "book test drives.")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------- hero
n_brands = len({m["brand"] for m in MODELS})
n_variants = sum(len(m["variants"]) for m in MODELS)
st.markdown(f"""
<div class="hero"><h1>Find your next car in <span>Pakistan</span></h1>
<p>Prices, comparisons, bank financing, spare parts and test drives, in English or Roman Urdu.</p>
<span class="chip gold">{len(MODELS)} models</span><span class="chip gold">{n_variants} variants</span>
<span class="chip gold">{n_brands} brands</span><span class="chip gold">8 banks</span>
<span class="chip">Prices as of Nov 2026</span><span class="chip">RAG + fine-tuned LLM + AI agent</span></div>
""", unsafe_allow_html=True)

chat_tab, showroom_tab, finance_tab, compare_tab, admin_tab = st.tabs(
    [":material/forum: Assistant", ":material/storefront: Showroom", ":material/account_balance: Financing",
     ":material/compare_arrows: Compare", ":material/admin_panel_settings: Admin"])

# ---------------------------------------------------------------- assistant
with chat_tab:
    if not online:
        st.warning(OFFLINE)
    if not ss.history:
        st.markdown("<p style='color:#8b95a9'>Assalam-o-Alaikum! Ask about any Toyota, Honda, Suzuki, KIA, Hyundai, "
                    "Changan or MG car, bank financing or spare parts, or pick an example from the sidebar.</p>",
                    unsafe_allow_html=True)
    last_mentioned: list[dict] = []
    for msg in ss.history:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/support_agent:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                mentioned = models_in_text(msg["content"], MODELS)
                last_mentioned = mentioned
                if mentioned:
                    st.markdown(mini_cards(mentioned), unsafe_allow_html=True)
                if msg.get("steps"):
                    tools_used = " → ".join(f"`{s['tool']}`" for s in msg["steps"])
                    with st.expander(f"Agent steps: {tools_used}", icon=":material/psychology:"):
                        for i, s in enumerate(msg["steps"], 1):
                            st.markdown(f"**{i}. `{s['tool']}`** `{s['args']}`")
                            st.code(s["result"][:1500], language=None)
    if last_mentioned and online:
        first = key_of(last_mentioned[0])
        st.caption("Follow up")
        cols = st.columns(4)
        cols[0].button(f"Bank options for {first}", icon=":material/account_balance:", width="stretch",
                       on_click=queue, args=(f"Bank financing options for the {first} for 3 years",))
        cols[1].button(f"Book a test drive", icon=":material/event:", width="stretch",
                       on_click=queue, args=(f"I want to book a test drive of the {first}",))
        cols[2].button("Open in Financing", icon=":material/calculate:", width="stretch",
                       on_click=set_finance, args=(first,))
        if len(last_mentioned) >= 2:
            second = key_of(last_mentioned[1])
            cols[3].button(f"Compare top two", icon=":material/compare_arrows:", width="stretch",
                           on_click=set_compare, args=(first, second))
    if prompt := st.chat_input("Ask about prices, comparisons, bank loans, parts or bookings...", disabled=not online):
        ask(prompt)
        st.rerun()

# ---------------------------------------------------------------- showroom
with showroom_tab:
    c1, c2, c3 = st.columns([2.2, 2, 1.4])
    search = c1.text_input("Search", placeholder="e.g. Corolla, hybrid, Suzuki", label_visibility="collapsed")
    lo, hi = c2.slider("Budget (lakh PKR)", 25, 170, (25, 170), step=5, label_visibility="collapsed",
                       format="%d lakh")
    sort = c3.selectbox("Sort", ["Price: low to high", "Price: high to low", "Name"], label_visibility="collapsed")
    brands = st.pills("Brand", sorted({m["brand"] for m in MODELS}), selection_mode="multi")
    f1, f2 = st.columns(2)
    body = f1.pills("Body type", ["hatchback", "sedan", "crossover", "suv", "mpv", "pickup"])
    fuel = f2.pills("Fuel", ["petrol", "hybrid", "electric", "diesel"])
    q = (search or "").lower().strip()
    shown = [m for m in MODELS
             if (not brands or m["brand"] in brands)
             and m["min_price"] <= hi * 100_000 and m["max_price"] >= lo * 100_000
             and (not body or body in m["body_types"]) and (not fuel or fuel in m["fuels"])
             and (not q or q in f"{m['brand']} {m['model']} {' '.join(m['fuels'])} {' '.join(m['body_types'])}".lower())]
    if sort == "Name":
        shown.sort(key=key_of)
    else:
        shown.sort(key=lambda m: m["min_price"], reverse=sort.startswith("Price: high"))
    st.caption(f"{len(shown)} of {len(MODELS)} models · {DISCLAIMER}")
    if not shown:
        st.info("No cars match these filters. Try widening the budget or clearing a filter.")
    cols = st.columns(3)
    for i, m in enumerate(shown):
        name = key_of(m)
        fav = name in ss.shortlist
        with cols[i % 3]:
            st.markdown(car_card(m, fav), unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.button("Saved" if fav else "Save", key=f"fav-{name}", width="stretch",
                      icon=":material/star:" if fav else ":material/star_outline:",
                      on_click=toggle_shortlist, args=(name,))
            b2.button("Finance", key=f"fin-{name}", width="stretch", icon=":material/calculate:",
                      on_click=set_finance, args=(name,))
            b3.button("Ask", key=f"ask-{name}", width="stretch", icon=":material/forum:",
                      disabled=not online, on_click=queue,
                      args=(f"Tell me about the {name}: all variants, prices and who it suits.",))
            with st.expander(f"{len(m['variants'])} variant{'s' if len(m['variants']) != 1 else ''}"):
                rows = "".join(f"<tr><td>{esc(v['variant'])}</td><td style='color:#d4af37'>Rs {v['price']:,}</td>"
                               f"<td>{esc(v['transmission'])}</td></tr>" for v in m["variants"])
                st.markdown(f"<table class='spec'><tr><th>Variant</th><th>Price</th><th>Gearbox</th></tr>{rows}</table>",
                            unsafe_allow_html=True)
                if m["features"]:
                    st.caption("Features: " + ", ".join(sorted(m["features"])))

# ---------------------------------------------------------------- financing
with finance_tab:
    st.markdown("#### Bank financing calculator")
    left, right = st.columns([1, 1.6])
    with left:
        car_name = st.selectbox("Car", NAMES, index=NAMES.index(ss.fin_car) if ss.fin_car in NAMES else 0)
        ss.fin_car = car_name
        m = BY_NAME[car_name]
        st.markdown(car_card(m), unsafe_allow_html=True)
        variant = st.selectbox("Variant", [f"{v['variant']} — Rs {v['price']:,}" for v in m["variants"]])
        v = m["variants"][[f"{x['variant']} — Rs {x['price']:,}" for x in m["variants"]].index(variant)]
        cc = int(v["engine_cc"]) if v["engine_cc"] else None
        terms = sbp_terms(v["price"], cc)
        years = st.segmented_control("Loan period (years)", YEAR_OPTIONS, default=min(3, terms["max_years"]),
                                     format_func=lambda y: f"{y} yr") or min(3, terms["max_years"])
        min_down = int(max(30, -(-terms["min_down"] * 100 // v["price"])))
        min_down = min(90, -(-min_down // 5) * 5)  # round up to the slider step
        if min_down > 30:
            st.info(f"SBP caps bank car loans at Rs 3,000,000 per person, so this car needs at least "
                    f"{min_down}% down (Rs {terms['min_down']:,}).", icon=":material/gavel:")
        down_pct = st.slider("Down payment (%)", min_down, 95, min_down, step=5)
        bank_type = st.pills("Bank type", ["Islamic", "Conventional"], selection_mode="multi")
    with right:
        qt = quote(v["price"], cc, years=years, down_pct=down_pct)
        for note in qt["notes"]:
            st.warning(note, icon=":material/gavel:")
        k1, k2, k3, k4 = st.columns(4)
        metric(k1, "Car price", f"Rs {fmt_pkr(qt['price'])}")
        metric(k2, "Down payment", f"Rs {fmt_pkr(qt['down'])}")
        metric(k3, "Bank loan", f"Rs {fmt_pkr(qt['loan'])}")
        best = min(qt["banks"], key=lambda r: r["emi_low"])
        metric(k4, f"Lowest monthly ({qt['years']} yr)", f"Rs {best['emi_low']:,}", gold=True)
        rows = [r for r in qt["banks"] if not bank_type or r["type"] in bank_type]
        st.write("")
        st.markdown(f"**Monthly instalment by bank** · {qt['years']} years")
        if rows:
            st.bar_chart({"Bank": [r["bank"] for r in rows],
                          "Lowest rate": [r["emi_low"] for r in rows],
                          "Highest rate": [r["emi_high"] for r in rows]},
                         x="Bank", y=["Lowest rate", "Highest rate"], stack=False, height=260,
                         color=["#d4af37", "#5b6b8c"], horizontal=True)
            st.dataframe(
                [{"Bank": r["bank"], "Product": r["product"], "Type": r["type"],
                  "Rate %": f"{r['rate_low']:g}–{r['rate_high']:g}", "Monthly from": r["emi_low"],
                  "Monthly to": r["emi_high"], "Total paid from": r["total_low"] + qt["down"]} for r in rows],
                hide_index=True, width="stretch",
                column_config={"Monthly from": st.column_config.NumberColumn(format="Rs %d"),
                               "Monthly to": st.column_config.NumberColumn(format="Rs %d"),
                               "Total paid from": st.column_config.NumberColumn(format="Rs %d")})
        st.markdown("**How the period changes your instalment**")
        bank_pick = st.selectbox("Rate to use", [f"{r['bank']} ({(r['rate_low'] + r['rate_high']) / 2:.1f}%)"
                                                 for r in qt["banks"]])
        rate = float(bank_pick.rsplit("(", 1)[1].rstrip("%)"))
        by_years = emi_by_years(v["price"], cc, down_pct=down_pct, rate_pct=rate)
        st.bar_chart({"Years": [f"{r['years']} yr" for r in by_years],
                      "Monthly instalment": [r["emi"] for r in by_years],
                      "Total markup": [r["markup"] for r in by_years]},
                     x="Years", y=["Monthly instalment", "Total markup"], stack=False, height=240,
                     color=["#d4af37", "#c0392b"])
        st.caption(f"Only the periods SBP allows for this car are shown (up to {terms['max_years']} years). "
                   "Longer loans lower the monthly instalment but raise the total markup.")
        with st.expander("Rules, rates and sources", icon=":material/info:"):
            st.markdown(SBP_NOTE)
            st.markdown("A draft government policy (AIDEP 2026–31) proposes 7-year loans, 15% down payment and a "
                        "Rs 10 million cap, but it is not in force yet.")
            st.markdown("Rates are indicative ranges from published 2026 bank comparisons and the SBP 1-year KIBOR "
                        "(12.19%, Sep 2026). Banks set the final rate; confirm before applying.")
            for r in qt["banks"]:
                st.markdown(f"- {r['bank']} {r['product']}: {r['basis']} ([source]({r['source']}))")
        st.button("Ask the assistant about this loan", icon=":material/forum:", disabled=not online,
                  on_click=queue, args=(f"Bank financing options for the {car_name} {v['variant']} for "
                                        f"{qt['years']} years with {down_pct}% down payment",))

# ---------------------------------------------------------------- compare
with compare_tab:
    c1, c2 = st.columns(2)
    a = c1.selectbox("First car", NAMES, index=NAMES.index(ss.cmp_a) if ss.cmp_a in NAMES else 0)
    b = c2.selectbox("Second car", NAMES, index=NAMES.index(ss.cmp_b) if ss.cmp_b in NAMES else 1)
    ss.cmp_a, ss.cmp_b = a, b
    ma, mb = BY_NAME[a], BY_NAME[b]
    c1.markdown(car_card(ma), unsafe_allow_html=True)
    c2.markdown(car_card(mb), unsafe_allow_html=True)

    def spec(m, key):
        if key == "Price range":
            return f"Rs {m['min_price']:,} – {m['max_price']:,}"
        if key == "Variants":
            return len(m["variants"])
        if key == "Engine":
            cc_list = sorted({x["engine_cc"] for x in m["variants"] if x["engine_cc"]})
            return ", ".join(f"{c} cc" for c in cc_list) or "Electric"
        if key == "Fuel":
            return ", ".join(sorted(m["fuels"]))
        if key == "Gearbox":
            return ", ".join(sorted({x["transmission"] for x in m["variants"]}))
        if key == "Seats":
            return ", ".join(str(s) for s in sorted(m["seats"]))
        if key == "Fuel average":
            return ", ".join(sorted({x["mileage"] + " km/l" for x in m["variants"] if x["mileage"]})) or "not listed"
        if key == "Max loan period":
            first_cc = next((x["engine_cc"] for x in m["variants"] if x["engine_cc"]), None)
            return f"{sbp_terms(m['min_price'], int(first_cc) if first_cc else None)['max_years']} years"
        return ""

    keys = ["Price range", "Variants", "Engine", "Fuel", "Gearbox", "Seats", "Fuel average", "Max loan period"]
    rows = "".join(f"<tr><th>{k}</th><td>{esc(spec(ma, k))}</td><td>{esc(spec(mb, k))}</td></tr>" for k in keys)
    st.markdown(f"<table class='spec'><tr><th></th><th>{esc(a)}</th><th>{esc(b)}</th></tr>{rows}</table>",
                unsafe_allow_html=True)
    st.write("")
    st.bar_chart({"Car": [a, b], "Starting price": [ma["min_price"], mb["min_price"]],
                  "Top variant": [ma["max_price"], mb["max_price"]]},
                 x="Car", y=["Starting price", "Top variant"], stack=False, height=230, color=["#d4af37", "#5b6b8c"])
    d1, d2, d3 = st.columns(3)
    d1.button(f"Finance {a}", icon=":material/calculate:", width="stretch", on_click=set_finance, args=(a,))
    d2.button(f"Finance {b}", icon=":material/calculate:", width="stretch", on_click=set_finance, args=(b,))
    d3.button("Which one suits me?", icon=":material/forum:", width="stretch", disabled=not online,
              on_click=queue, args=(f"Compare {a} and {b}. Which is better value for a Pakistani family?",))

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
            metric(m1, "Total bookings", str(len(bks)))
            metric(m2, "Test drives", str(sum(x["kind"] == "test_drive" for x in bks)))
            metric(m3, "Service bookings", str(sum(x["kind"] == "service" for x in bks)))
            metric(m4, "Escalation tickets", str(len(tks)), gold=True)
            st.write("")
            kind = st.segmented_control("Show", ["All", "Test drives", "Services"], default="All")
            shown_b = [x for x in bks if kind in (None, "All") or
                       (kind == "Test drives" and x["kind"] == "test_drive") or (kind == "Services" and x["kind"] == "service")]
            st.markdown("**Bookings made by the agent**")
            st.dataframe(shown_b, width="stretch", hide_index=True)
            if bks:
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=list(bks[0]))
                w.writeheader()
                w.writerows(bks)
                st.download_button("Download bookings (CSV)", buf.getvalue(), "bookings.csv", "text/csv",
                                   icon=":material/download:")
            st.markdown("**Conversations handed to staff**")
            st.dataframe(tks, width="stretch", hide_index=True)
        except BackendOffline:
            st.error(OFFLINE)
    elif pw:
        st.error("Wrong password.")
