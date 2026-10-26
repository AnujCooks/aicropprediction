"""
Generates a realistic synthetic dataset mimicking Agmarknet (mandi prices) +
ICAR/state agri department (yield) + IMD (rainfall) records for Indian
agriculture, since live government portals aren't reachable from this
environment. Structure and value ranges are modeled on real published
statistics so the ML pipeline and dashboard behave the way they would on
real data. Swap load_data() in app.py with a real CSV any time.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

STATES = {
    "Punjab":        {"crops": ["Wheat", "Rice", "Cotton"],            "base_rain": 650,  "soil": "Alluvial"},
    "Uttar Pradesh": {"crops": ["Wheat", "Rice", "Sugarcane"],         "base_rain": 900,  "soil": "Alluvial"},
    "Maharashtra":   {"crops": ["Cotton", "Soybean", "Jowar"],         "base_rain": 1100, "soil": "Black"},
    "Madhya Pradesh":{"crops": ["Soybean", "Wheat", "Gram"],           "base_rain": 1050, "soil": "Black"},
    "Karnataka":     {"crops": ["Ragi", "Maize", "Cotton"],            "base_rain": 1200, "soil": "Red"},
    "Rajasthan":     {"crops": ["Bajra", "Mustard", "Wheat"],          "base_rain": 450,  "soil": "Arid"},
    "West Bengal":   {"crops": ["Rice", "Jute", "Potato"],             "base_rain": 1600, "soil": "Alluvial"},
    "Bihar":         {"crops": ["Rice", "Wheat", "Maize"],             "base_rain": 1150, "soil": "Alluvial"},
    "Gujarat":       {"crops": ["Cotton", "Groundnut", "Bajra"],       "base_rain": 800,  "soil": "Black"},
    "Andhra Pradesh":{"crops": ["Rice", "Cotton", "Chilli"],           "base_rain": 950,  "soil": "Red"},
}

# Approx realistic base yield (quintal/hectare) and base price (Rs/quintal)
CROP_PROFILE = {
    "Wheat":     {"base_yield": 34, "base_price": 2100, "price_vol": 0.10},
    "Rice":      {"base_yield": 27, "base_price": 1950, "price_vol": 0.09},
    "Cotton":    {"base_yield": 15, "base_price": 6300, "price_vol": 0.18},
    "Sugarcane": {"base_yield": 700,"base_price": 315,  "price_vol": 0.06},
    "Soybean":   {"base_yield": 12, "base_price": 4200, "price_vol": 0.16},
    "Jowar":     {"base_yield": 10, "base_price": 2800, "price_vol": 0.14},
    "Gram":      {"base_yield": 11, "base_price": 5200, "price_vol": 0.15},
    "Ragi":      {"base_yield": 16, "base_price": 3400, "price_vol": 0.12},
    "Maize":     {"base_yield": 28, "base_price": 1950, "price_vol": 0.13},
    "Bajra":     {"base_yield": 13, "base_price": 2350, "price_vol": 0.14},
    "Mustard":   {"base_yield": 14, "base_price": 5300, "price_vol": 0.15},
    "Jute":      {"base_yield": 25, "base_price": 4700, "price_vol": 0.11},
    "Potato":    {"base_yield": 220,"base_price": 1200, "price_vol": 0.25},
    "Groundnut": {"base_yield": 17, "base_price": 5800, "price_vol": 0.17},
    "Chilli":    {"base_yield": 18, "base_price": 12000,"price_vol": 0.22},
}

SEASONS = ["Kharif", "Rabi"]
YEARS = list(range(2015, 2025))


def generate_yield_dataset(n_extra_noise_rows=0):
    rows = []
    for year in YEARS:
        # national-level year effects: monsoon quality, input cost inflation
        monsoon_factor = np.random.normal(1.0, 0.08)
        for state, meta in STATES.items():
            for crop in meta["crops"]:
                profile = CROP_PROFILE[crop]
                for season in SEASONS:
                    if crop == "Wheat" and season == "Kharif":
                        continue
                    if crop in ["Rice", "Cotton", "Soybean", "Jowar", "Bajra", "Groundnut", "Chilli"] and season == "Rabi":
                        continue

                    rainfall = max(150, np.random.normal(meta["base_rain"], meta["base_rain"] * 0.18) * monsoon_factor)
                    avg_temp = np.random.normal(27 if season == "Kharif" else 20, 2.5)
                    fertilizer_kg_ha = np.random.normal(140, 25)
                    area_hectare = np.random.uniform(5000, 250000)
                    pesticide_use = np.random.uniform(0.5, 3.5)
                    irrigation_pct = np.clip(np.random.normal(55, 20), 5, 100)

                    # yield model: rainfall & fertilizer help up to a point, extreme rainfall hurts
                    rain_dev = (rainfall - meta["base_rain"]) / meta["base_rain"]
                    rain_effect = -0.35 * (rain_dev ** 2) + 0.15 * rain_dev
                    fert_effect = 0.10 * (fertilizer_kg_ha - 140) / 140
                    irrigation_effect = 0.08 * (irrigation_pct - 55) / 55
                    year_trend = 0.012 * (year - 2015)  # tech/productivity improvement over years

                    yield_multiplier = 1 + rain_effect + fert_effect + irrigation_effect + year_trend
                    yield_multiplier *= np.random.normal(1.0, 0.05)
                    yield_qtl_ha = max(1, profile["base_yield"] * yield_multiplier)

                    rows.append({
                        "year": year,
                        "state": state,
                        "crop": crop,
                        "season": season,
                        "soil_type": meta["soil"],
                        "rainfall_mm": round(rainfall, 1),
                        "avg_temp_c": round(avg_temp, 1),
                        "fertilizer_kg_per_ha": round(fertilizer_kg_ha, 1),
                        "pesticide_l_per_ha": round(pesticide_use, 2),
                        "irrigation_coverage_pct": round(irrigation_pct, 1),
                        "area_hectare": round(area_hectare, 0),
                        "yield_quintal_per_hectare": round(yield_qtl_ha, 2),
                    })
    df = pd.DataFrame(rows)
    return df


def generate_price_timeseries():
    """Monthly mandi price series per (state, crop) for time-series forecasting."""
    records = []
    months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    for state, meta in STATES.items():
        for crop in meta["crops"]:
            profile = CROP_PROFILE[crop]
            base = profile["base_price"]
            vol = profile["price_vol"]
            price = base * np.random.uniform(0.9, 1.1)
            trend = np.random.uniform(0.001, 0.004)  # slow inflationary drift
            for i, m in enumerate(months):
                seasonal = 1 + 0.06 * np.sin(2 * np.pi * (m.month / 12) + hash(crop) % 6)
                shock = np.random.normal(0, vol * 0.3)
                price = price * (1 + trend + shock) * (0.995 + 0.01 * seasonal)
                price = max(price, base * 0.4)
                records.append({
                    "date": m,
                    "state": state,
                    "crop": crop,
                    "modal_price_rs_per_quintal": round(price, 1),
                })
    return pd.DataFrame(records)


if __name__ == "__main__":
    yield_df = generate_yield_dataset()
    price_df = generate_price_timeseries()
    yield_df.to_csv("yield_data.csv", index=False)
    price_df.to_csv("price_data.csv", index=False)
    print("yield_data.csv:", yield_df.shape)
    print("price_data.csv:", price_df.shape)
    print(yield_df.head())
    print(price_df.head())
