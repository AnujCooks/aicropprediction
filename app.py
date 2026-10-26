import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Google Gemini (free tier — no credit card required)
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KrishiMitra AI — Crop Yield & Price Analytics",
    page_icon="🌾",
    layout="wide"
)

BASE = Path(__file__).parent

# Model used for the AI Agriculture Assistant.
# gemini-2.5-flash is on Google's free tier (no credit card required),
# with generous daily rate limits — good fit for a hackathon demo.
GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# LOAD ML MODELS
# ============================================================

@st.cache_resource
def load_yield_model():
    model_path = BASE / "yield_model.pkl"

    if not model_path.exists():
        st.error(
            f"❌ yield_model.pkl not found.\n\n"
            f"Expected location:\n{model_path}"
        )
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_price_model():
    model_path = BASE / "price_model.pkl"

    if not model_path.exists():
        st.error(
            f"❌ price_model.pkl not found.\n\n"
            f"Expected location:\n{model_path}"
        )
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_yield_df():
    file_path = BASE / "yield_data.csv"

    if not file_path.exists():
        st.error(f"❌ yield_data.csv not found:\n{file_path}")
        st.stop()

    return pd.read_csv(file_path)


@st.cache_data
def load_price_df():
    file_path = BASE / "price_data.csv"

    if not file_path.exists():
        st.error(f"❌ price_data.csv not found:\n{file_path}")
        st.stop()

    return pd.read_csv(file_path, parse_dates=["date"])


# Load everything
yield_artifact = load_yield_model()
price_artifact = load_price_model()
yield_df = load_yield_df()
price_df = load_price_df()


# ============================================================
# STATE-CROP MAP
# ============================================================

STATE_CROP_MAP = {}

for s in yield_artifact["options"]["state"]:
    STATE_CROP_MAP[s] = sorted(
        yield_df.loc[
            yield_df["state"] == s,
            "crop"
        ].unique().tolist()
    )


# ============================================================
# GEMINI AI ASSISTANT
# ============================================================

def get_api_key():
    """
    Looks for the Gemini API key in this order:
    1. GEMINI_API_KEY / GOOGLE_API_KEY environment variable
    2. st.secrets["GEMINI_API_KEY"] (Streamlit Cloud / .streamlit/secrets.toml)
    Get a free key (no credit card) at https://aistudio.google.com/apikey
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def get_gemini_client():
    if not GEMINI_AVAILABLE:
        return None

    api_key = get_api_key()

    if not api_key or api_key.strip() == "":
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None


def ask_ai_assistant(user_question, context=None):
    client = get_gemini_client()

    if client is None:
        return (
            "⚠️ AI Assistant is not configured.\n\n"
            "Make sure:\n"
            "1. `pip install google-genai` is done.\n"
            "2. A **free** Gemini API key is set in `.streamlit/secrets.toml` "
            "as `GEMINI_API_KEY = \"...\"` (get one at "
            "https://aistudio.google.com/apikey — no credit card needed), "
            "or as a `GEMINI_API_KEY` environment variable.\n"
            "3. The app was restarted after setting the key."
        )

    system_prompt = """
You are KrishiMitra AI, an intelligent agriculture assistant.

Your job is to help farmers and agriculture students with:

1. Crop selection
2. Crop yield
3. Soil
4. Irrigation
5. Fertilizer
6. Pesticide
7. Rainfall
8. Temperature
9. Crop diseases
10. Mandi prices
11. Agricultural economics
12. Farming best practices
13. Crop planning
14. Harvest planning
15. General agriculture questions

Important rules:

- Answer in simple language.
- User may ask in Hindi, Hinglish or English.
- Reply in the same language as the user whenever possible.
- Give practical and easy-to-understand answers.
- Do not claim that an ML prediction is guaranteed.
- Clearly distinguish between model prediction and general agricultural advice.
- For pesticide/fertilizer recommendations, encourage following the product label and local agricultural expert guidance.
- If information is uncertain, say so.
- Do not invent live mandi prices.
- Do not claim access to live weather unless live weather data is actually provided.
- Keep answers structured using headings and bullet points when useful.
"""

    if context:
        system_prompt += f"""

Current KrishiMitra application context:

{context}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_question,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        return response.text

    except Exception as e:
        return (
            "❌ AI Assistant Error\n\n"
            f"{str(e)}\n\n"
            "Please check your GEMINI_API_KEY, model name, and internet connection. "
            "If this says rate-limited, the free tier has a daily cap — wait a bit and retry."
        )


# ============================================================
# ML FUNCTIONS
# ============================================================

def predict_yield(
    state, crop, season, soil_type,
    rainfall, temp, fert, pesticide, irrigation, area
):
    row = pd.DataFrame([{
        "state": state,
        "crop": crop,
        "season": season,
        "soil_type": soil_type,
        "year": 2026,
        "rainfall_mm": rainfall,
        "avg_temp_c": temp,
        "fertilizer_kg_per_ha": fert,
        "pesticide_l_per_ha": pesticide,
        "irrigation_coverage_pct": irrigation,
        "area_hectare": area,
    }])

    pred = yield_artifact["pipeline"].predict(row)[0]
    return max(float(pred), 0)


def forecast_price(state, crop, months_ahead=6):
    key = (state, crop)

    if key not in price_artifact["models"]:
        return None, None

    model = price_artifact["models"][key]
    hist = price_artifact["histories"][key].copy()

    series = list(hist["modal_price_rs_per_quintal"].values)
    last_date = hist["date"].iloc[-1]

    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=months_ahead,
        freq="MS"
    )

    preds = []
    for fd in future_dates:
        feat = {
            "lag_1": series[-1],
            "lag_2": series[-2],
            "lag_3": series[-3],
            "lag_6": series[-6],
            "lag_12": series[-12],
            "rolling_mean_3": np.mean(series[-3:]),
            "rolling_std_3": np.std(series[-3:]),
            "month": fd.month,
        }
        X = pd.DataFrame([feat])
        p = float(model.predict(X)[0])
        preds.append(p)
        series.append(p)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "modal_price_rs_per_quintal": preds,
        "type": "Forecast"
    })

    hist_tagged = hist.copy()
    hist_tagged["type"] = "Historical"

    return hist_tagged, forecast_df


def advisory_text(state, crop, predicted_yield, hist_df, forecast_df):
    avg_hist_price = hist_df["modal_price_rs_per_quintal"].tail(12).mean()

    forecast_peak_idx = forecast_df["modal_price_rs_per_quintal"].idxmax()
    forecast_peak_month = forecast_df.loc[forecast_peak_idx, "date"].strftime("%B %Y")
    forecast_peak_price = forecast_df.loc[forecast_peak_idx, "modal_price_rs_per_quintal"]

    price_trend_pct = (
        (forecast_df["modal_price_rs_per_quintal"].iloc[-1] - avg_hist_price)
        / avg_hist_price * 100
    )

    lines = []
    lines.append(
        f"**Expected yield** for {crop} in {state}: "
        f"**{predicted_yield:.1f} quintal/hectare**."
    )

    if price_trend_pct > 5:
        lines.append(
            f"Prices are **trending upward** "
            f"({price_trend_pct:+.1f}% vs last 12-month average). "
            f"Model expects the best mandi price around "
            f"**{forecast_peak_month}** (₹{forecast_peak_price:,.0f}/quintal)."
        )
    elif price_trend_pct < -5:
        lines.append(
            f"Prices are **trending downward** "
            f"({price_trend_pct:+.1f}% vs last 12-month average). "
            f"Consider comparing available procurement/MSP options "
            f"and current local market prices before selling."
        )
    else:
        lines.append(
            f"Prices are expected to stay **relatively stable** "
            f"(~{price_trend_pct:+.1f}% vs last 12-month average)."
        )

    return "\n\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 KrishiMitra AI")
st.sidebar.caption("Data-driven crop yield & price advisory for farmers")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Advisory Dashboard",
        "🤖 AI Agriculture Assistant",
        "📊 Yield Model Insights",
        "💰 Price Trends Explorer",
        "ℹ️ About / Model Info"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built for Smart India Hackathon**\n\n"
    "Theme: Agriculture, FoodTech & Rural Development\n\n"
    "ML Models: Random Forest + Gradient Boosting/XGBoost"
)


# ============================================================
# PAGE 1 — ADVISORY DASHBOARD
# ============================================================

if page == "🏠 Advisory Dashboard":
    st.title("🌾 Crop Yield & Price Advisory")
    st.caption(
        "Enter your farm details to get an AI-based "
        "yield estimate and 6-month mandi price forecast."
    )

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("🌱 Farm Inputs")

        state = st.selectbox("State", yield_artifact["options"]["state"])
        crop = st.selectbox("Crop", STATE_CROP_MAP[state])
        season = st.selectbox("Season", yield_artifact["options"]["season"])
        soil_type = st.selectbox(
            "Soil Type",
            yield_artifact["options"]["soil_type"],
            index=yield_artifact["options"]["soil_type"].index(
                yield_df.loc[yield_df["state"] == state, "soil_type"].iloc[0]
            )
        )

        rainfall = st.slider("Expected Rainfall (mm)", 100, 2200, 900)
        temp = st.slider("Avg Temperature (°C)", 10, 40, 27)
        fert = st.slider("Fertilizer Use (kg/hectare)", 20, 300, 140)
        pesticide = st.slider("Pesticide Use (L/hectare)", 0.0, 5.0, 1.5)
        irrigation = st.slider("Irrigation Coverage (%)", 0, 100, 55)
        area = st.number_input("Farm Area (hectare)", min_value=0.1, value=2.0, step=0.1)

        run = st.button("🔮 Get Advisory", type="primary", use_container_width=True)

    with col2:
        if run:
            pred_yield = predict_yield(
                state, crop, season, soil_type,
                rainfall, temp, fert, pesticide, irrigation, area
            )
            hist, fcst = forecast_price(state, crop, months_ahead=6)

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Yield", f"{pred_yield:.1f} qtl/ha")
            m2.metric("Estimated Total Produce", f"{pred_yield * area:.0f} quintal")
            if fcst is not None:
                m3.metric(
                    "6-mo Forecast Peak Price",
                    f"₹{fcst['modal_price_rs_per_quintal'].max():,.0f}/qtl"
                )

            if hist is not None:
                combined = pd.concat([hist.tail(24), fcst], ignore_index=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=combined.loc[combined["type"] == "Historical", "date"],
                    y=combined.loc[combined["type"] == "Historical", "modal_price_rs_per_quintal"],
                    mode="lines", name="Historical Price"
                ))
                fig.add_trace(go.Scatter(
                    x=combined.loc[combined["type"] == "Forecast", "date"],
                    y=combined.loc[combined["type"] == "Forecast", "modal_price_rs_per_quintal"],
                    mode="lines+markers", name="Forecast"
                ))
                fig.update_layout(
                    title=f"{crop} Mandi Price — {state}",
                    xaxis_title="Month",
                    yaxis_title="Modal Price (₹/quintal)",
                    height=380
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Advisory")
                st.info(advisory_text(state, crop, pred_yield, hist, fcst))
        else:
            st.info("Fill in the farm details and click **Get Advisory**.")


# ============================================================
# PAGE 2 — AI ASSISTANT
# ============================================================

elif page == "🤖 AI Agriculture Assistant":
    st.title("🤖 KrishiMitra AI Assistant")
    st.caption(
        "Ask questions about crops, farming, soil, yield, "
        "fertilizer, irrigation, mandi prices and agriculture."
    )

    # API status
    client_ready = GEMINI_AVAILABLE and get_gemini_client() is not None

    if client_ready:
        st.success("🟢 AI Assistant is connected and ready. (Free Gemini API tier)")
    else:
        if not GEMINI_AVAILABLE:
            reason = "The `google-genai` package isn't installed. Run `pip install google-genai`."
        elif not get_api_key():
            reason = (
                "No Gemini API key found. Get a **free** key (no credit card) at "
                "https://aistudio.google.com/apikey, then add it to "
                "`.streamlit/secrets.toml` as `GEMINI_API_KEY = \"...\"` and restart the app."
            )
        else:
            reason = "Could not initialize the Gemini client — check the key value."
        st.warning(f"🟡 AI Assistant is not configured.\n\n{reason}")

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content":
                    "🌾 Namaste! Main KrishiMitra AI hoon. "
                    "Aap mujhse farming, crop, soil, fertilizer, "
                    "yield aur mandi price ke baare mein kuch bhi pooch sakte hain."
            }
        ]

    # Example prompts
    st.markdown("### 💡 Try asking")
    prompt_cols = st.columns(4)
    example_prompts = [
        "Which crop is best for low rainfall?",
        "गेहूं के लिए कौन सी मिट्टी अच्छी है?",
        "How can I improve crop yield?",
        "What is the importance of irrigation?"
    ]

    for i, prompt in enumerate(example_prompts):
        if prompt_cols[i].button(prompt, use_container_width=True):
            st.session_state.selected_prompt = prompt

    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    selected_prompt = st.session_state.pop("selected_prompt", None)
    user_prompt = st.chat_input("Ask KrishiMitra anything about agriculture...")

    if selected_prompt:
        user_prompt = selected_prompt

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        context = f"""
Available states:
{list(yield_artifact["options"]["state"])}

Available crops:
{sorted(yield_df["crop"].unique().tolist())}

The application has:
- Crop yield prediction model
- Mandi price forecasting model
- Historical crop price data
- Yield data
"""

        with st.chat_message("assistant"):
            with st.spinner("🌾 KrishiMitra AI is thinking..."):
                answer = ask_ai_assistant(user_prompt, context)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "🌾 Chat cleared. Aap apna question pooch sakte hain."
            }
        ]
        st.rerun()


# ============================================================
# PAGE 3 — YIELD MODEL INSIGHTS
# ============================================================

elif page == "📊 Yield Model Insights":
    st.title("📊 Yield Prediction Model — Insights")

    metrics = yield_artifact["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("MAE", f"{metrics['mae']:.2f} qtl/ha")
    c3.metric("RMSE", f"{metrics['rmse']:.2f} qtl/ha")

    st.markdown("#### Feature Importance")
    pipe = yield_artifact["pipeline"]
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        yield_artifact["categorical_features"]
    )
    all_names = list(cat_names) + yield_artifact["numeric_features"]
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": all_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    fig = px.bar(
        imp_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Greens",
        title="Top 15 Features Driving Yield Predictions"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Yield by Crop & State")
    pivot = yield_df.groupby(["crop", "state"])["yield_quintal_per_hectare"].mean().reset_index()
    fig2 = px.density_heatmap(
        pivot, x="state", y="crop", z="yield_quintal_per_hectare",
        color_continuous_scale="YlGn", title="Avg Yield (qtl/ha) Heatmap"
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Rainfall vs Yield")
    fig3 = px.scatter(
        yield_df, x="rainfall_mm", y="yield_quintal_per_hectare", color="crop",
        trendline="ols", opacity=0.6, title="Rainfall vs Yield by Crop"
    )
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# PAGE 4 — PRICE TRENDS
# ============================================================

elif page == "💰 Price Trends Explorer":
    st.title("💰 Mandi Price Trends Explorer")

    col1, col2 = st.columns(2)
    with col1:
        state_sel = st.selectbox("State", sorted(price_df["state"].unique()))
    with col2:
        crop_options = sorted(price_df.loc[price_df["state"] == state_sel, "crop"].unique())
        crop_sel = st.selectbox("Crop", crop_options)

    filtered = price_df[
        (price_df["state"] == state_sel) & (price_df["crop"] == crop_sel)
    ].sort_values("date")

    fig = px.line(
        filtered, x="date", y="modal_price_rs_per_quintal",
        title=f"{crop_sel} Price History — {state_sel}", markers=False
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    key = (state_sel, crop_sel)
    if key in price_artifact["metrics"]:
        st.caption(
            f"Model holdout MAPE for this crop-state pair: "
            f"**{price_artifact['metrics'][key]:.2%}**"
        )

    st.markdown("#### Compare Multiple Crops (same state)")
    multi_crops = st.multiselect("Select crops to compare", crop_options, default=crop_options[:2])
    if multi_crops:
        multi_df = price_df[
            (price_df["state"] == state_sel) & (price_df["crop"].isin(multi_crops))
        ]
        fig2 = px.line(
            multi_df, x="date", y="modal_price_rs_per_quintal", color="crop",
            title=f"Price Comparison — {state_sel}"
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

else:
    st.title("ℹ️ About KrishiMitra AI")

    st.markdown(f"""
# 🌾 KrishiMitra AI

### Crop Yield & Price Advisory Platform

**Problem it solves**

Small and marginal farmers often lack timely,
data-driven guidance on expected yield and
market prices.

### What this prototype does

- 🌱 Predicts crop yield
- 💰 Forecasts mandi prices
- 📊 Shows price trends
- 📈 Shows ML model insights
- 🤖 Provides an AI agriculture assistant
- 🧑‍🌾 Gives simple farming guidance

### Machine Learning

**Yield Model**

Random Forest Regressor

R² Score: **{yield_artifact['metrics']['r2']:.3f}**

**Price Model**

Gradient Boosting / XGBoost

Average holdout MAPE: **{price_artifact['avg_mape']:.2%}**

### AI Assistant

The AI Assistant allows users to ask questions
in English, Hindi or Hinglish about agriculture.
Powered by the free-tier Google Gemini API (`{GEMINI_MODEL}`).

Example:

> "गेहूं की फसल में उत्पादन कैसे बढ़ाएं?"

> "Which crop is best for low rainfall?"

> "What fertilizer is suitable for wheat?"

### Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Google Gemini API (free tier)

### Future Scope

- Live weather API
- Real-time mandi prices
- Hindi/Punjabi regional language support
- WhatsApp/SMS advisory
- Crop disease detection
- Voice-based farmer assistant
- Government scheme information
- Crop recommendation system

### ⚠️ Important

ML predictions are estimates and should not be
treated as guaranteed outcomes.

For pesticide, fertilizer and crop disease
decisions, farmers should also consult local
agriculture experts and follow product labels.
""")