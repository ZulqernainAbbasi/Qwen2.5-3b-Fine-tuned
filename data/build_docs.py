"""Generate one markdown doc per model from cars.csv plus short editorial notes.

Run: python -m data.build_docs
"""
import config
from data.validate import load_cars, model_doc_name

NOTES = {
    ("Suzuki", "Alto"): "Pakistan's best-selling entry-level 660cc city hatchback. Known for low running cost, cheap parts, a large dealer network and quick resale. Updated in April 2025 with airbags and ABS.",
    ("Suzuki", "Every"): "A 660cc micro-van, mainly bought for small business, school runs and cargo use.",
    ("Suzuki", "Cultus"): "1.0-litre five-door hatchback, a step up from the Alto in space and power. AGS (auto gear shift) variant available.",
    ("Suzuki", "Swift"): "Suzuki's most premium hatchback in Pakistan, with a 1.2-litre engine and a CVT option. Sportier styling than the Cultus.",
    ("Suzuki", "Fronx"): "Pakistan's first locally assembled Suzuki crossover (launched 6 May 2026) with a 1.5-litre K-series engine. The top GLX trim adds mild-hybrid tech, 360 camera and cruise control.",
    ("Toyota", "Yaris"): "Toyota's entry sedan in Pakistan, positioned below the Corolla. Offered with 1.3 and 1.5-litre engines in manual and CVT.",
    ("Toyota", "Corolla"): "Pakistan's most trusted family sedan, with strong resale value. The facelifted 2027 model got introductory prices for vehicles invoiced 16 Oct to 31 Dec 2026.",
    ("Toyota", "Corolla Cross"): "Toyota's compact crossover with 1.8 petrol and 1.8 hybrid (HEV) options. Hybrid prices were cut on 13 Sep 2026.",
    ("Toyota", "Hilux"): "The Hilux E (standard) is the workhorse pickup below the Revo line.",
    ("Toyota", "Hilux Revo"): "Popular 2.8-litre diesel double-cab pickup used for both work and family. The Rocco is the sporty variant.",
    ("Toyota", "Fortuner"): "Body-on-frame 7-seat SUV, popular for long-distance and rough-road driving.",
    ("Honda", "City"): "Honda's compact sedan. Honda keeps the base City under Rs 5 million so it stays in the lower sales-tax bracket (18% instead of 25%).",
    ("Honda", "Civic"): "Honda's premium sedan with a 1.5-litre turbo engine, offered in Standard, Oriel and RS trims.",
    ("Honda", "BR-V"): "A 7-seat MPV-crossover based on the City platform, aimed at larger families.",
    ("Honda", "HR-V"): "Compact crossover. The e:HEV is a two-motor hybrid. Both HR-V variants got limited-time price cuts on 13 Sep 2026.",
    ("KIA", "Picanto"): "KIA's 1.0-litre city hatchback with an automatic gearbox.",
    ("KIA", "Stonic"): "Subcompact crossover. The new 1.0L Turbo version opened for booking on 1 Oct 2026 in X and S trims.",
    ("KIA", "Sportage L"): "KIA's popular compact SUV. The 2.0L FWD is in limited stock. Hybrid (HEV) variants had a promotional offer running 3 to 30 Nov 2026.",
    ("KIA", "Sorento"): "Large 7-seat SUV. The 3.5L V6 petrol is the non-hybrid option.",
    ("Hyundai", "Elantra"): "Locally assembled hybrid sedan with a 1.6-litre GDi engine and an electric motor. One of the most fuel-efficient sedans on sale.",
    ("Hyundai", "Tucson"): "Fourth-generation Tucson, sold as a 1.6 turbo hybrid in Smart FWD and Signature AWD trims.",
    ("Hyundai", "Santa Fe"): "Hyundai's locally assembled 7-seat hybrid SUV.",
    ("Hyundai", "Sonata"): "Mid-size premium sedan. The N Line is the sporty turbocharged trim.",
    ("Changan", "Alsvin"): "Changan's affordable 1.5-litre sedan, priced against hatchbacks. All Alsvin variants were cut by Rs 400,000 in Aug 2026.",
    ("Changan", "Karvaan"): "A 7-seat 1.2-litre mini-van, an affordable option for large families and commercial use.",
    ("Changan", "Oshan X7"): "1.5 turbo SUV in 5 and 7-seat layouts. Prices were discounted in Nov 2026.",
    ("MG", "ZS"): "MG's compact SUV, offered as a petrol (Vibe) and as a Hybrid+.",
    ("MG", "HS"): "Second-generation HS launched in April 2026, sold in hybrid versions only.",
    ("MG", "MG4"): "Fully electric hatchback with a 43 kWh battery and about 316 km of range.",
    ("MG", "Binguo EV"): "Small electric city car. Price cut by Rs 300,000 in May 2026.",
    ("MG", "ZS EV"): "Electric compact SUV. The Long Range version has about 505 km of range.",
}


def render(brand: str, model: str, rows) -> str:
    lines = [f"# {brand} {model}", f"Source: {rows.iloc[0].source_url}", ""]
    lines.append(NOTES.get((brand, model), ""))
    lo, hi = rows.price_pkr.min(), rows.price_pkr.max()
    lines += ["", f"Price range (ex-factory, Nov 2026): Rs {lo:,} to Rs {hi:,}.", "", "## Variants"]
    for r in rows.itertuples():
        eng = f"{r.engine_cc} cc" if str(r.engine_cc) not in ("<NA>", "nan") else "electric motor"
        bits = [f"Rs {r.price_pkr:,}", eng, r.fuel, r.transmission, f"{r.seats} seats"]
        if r.mileage_kmpl:
            bits.append(f"{r.mileage_kmpl} km/l")
        line = f"- {r.variant}: " + ", ".join(bits)
        if r.key_features:
            line += f". Features: {r.key_features.replace(';', ', ')}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    cars = load_cars()
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for (brand, model), rows in cars.groupby(["brand", "model"], sort=False):
        (config.DOCS_DIR / model_doc_name(brand, model)).write_text(render(brand, model, rows), encoding="utf-8")
        n += 1
    print(f"wrote {n} model docs")
    return n


if __name__ == "__main__":
    main()
