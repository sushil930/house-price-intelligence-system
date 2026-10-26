"""Streamlit dashboard.

Loads the real Bengaluru dataset, runs the same cleaning,
preprocessing and modelling code as the pipeline, shows three key
charts, and offers a sidebar price-prediction form with a styled
result card.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from house_price import cleaning, eda_stats, loading, modeling, preprocessing

st.set_page_config(page_title="House Price Intelligence", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(135deg, #1f2a44 0%, #3b2d5c 100%);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: 0.85; }
    .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        flex: 1; padding: 1rem 1.2rem; border-radius: 14px;
        background: #1a1f2e; border: 1px solid #2a3248;
    }
    .kpi-label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; }
    .result-card {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(135deg, #0f5132 0%, #146c43 100%);
        margin-top: 1rem;
    }
    .result-card .price { font-size: 2rem; font-weight: 800; }
    .result-card .note { opacity: 0.85; margin-top: 0.3rem; }
    div.stButton > button {
        background: #6c5ce7; color: white; border: none;
        border-radius: 10px; font-weight: 600; width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🏠 House Price Intelligence</h1>
      <p>Bengaluru house prices, from raw CSV to a trained regression
      model — covering all five units of the Data Science using Python
      (CAPCTD3) syllabus.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_train():
    """Run cleaning + preprocessing + modelling on the real data."""
    raw = loading.load_csv(
        Path(__file__).parent / "data" / "Bengaluru_House_Data.csv"
    )
    clean, _ = cleaning.phase1_clean(raw)
    with_pps = preprocessing.add_price_per_sqft(clean)
    filtered, _ = preprocessing.apply_outlier_filter(with_pps)
    encoded, _ = preprocessing.encode_features(filtered)
    eda = eda_stats.engineer_features(encoded)
    X, y, features, _ = modeling.build_features(eda)
    model = modeling.train_multiple(X, y)
    return eda, X, y, features, model


eda, X, y, features, model = load_and_train()

st.subheader("At a glance")
st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Listings (cleaned)</div>
        <div class="kpi-value">{len(eda):,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mean price</div>
        <div class="kpi-value">{eda["price"].mean():.2f} lakh</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Median price</div>
        <div class="kpi-value">{eda["price"].median():.2f} lakh</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Charts")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor("#0e1117")
for ax in axes:
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
axes[0].hist(eda["price"], bins=20, color="#6c5ce7")
axes[0].set_title("Price distribution")
axes[0].set_xlabel("Price (lakh)")
axes[1].scatter(eda["total_sqft"], eda["price"], s=8, alpha=0.4, color="#00b894")
axes[1].set_title("Sqft vs price")
axes[1].set_xlabel("Total sqft")
axes[1].set_ylabel("Price (lakh)")
med = eda.groupby("bhk")["price"].median().sort_index()
axes[2].bar(med.index.astype(str), med.values, color="#fdcb6e")
axes[2].set_title("Median price by BHK")
axes[2].set_xlabel("BHK")
fig.tight_layout()
st.pyplot(fig)

st.sidebar.header("🔍 Predict a price")
st.sidebar.write("Enter the property details and hit Predict.")
with st.sidebar.form("predict"):
    total_sqft = st.number_input("Total sqft", min_value=100.0, value=1200.0)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)
    tier = st.selectbox(
        "Location tier (0 = budget … 3 = luxury)",
        options=sorted(eda["location_tier"].unique().tolist()),
    )
    super_built = st.checkbox("Super built-up area", value=True)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "total_sqft": total_sqft,
        "bath": float(bath),
        "bhk": float(bhk),
        "balcony": float(balcony),
        "location_tier": float(tier),
        "room_density": float(bhk) / (total_sqft / 1000),
        "is_premium": 1.0 if tier >= 2 else 0.0,
    }
    for col in features:
        if col.startswith("area_type_"):
            row[col] = 1.0 if super_built else 0.0
    import pandas as pd

    X_new = pd.DataFrame([[row[c] for c in features]], columns=features)
    price = float(model.predict(X_new)[0])
    st.markdown(
        f"""
        <div class="result-card">
          <div class="price">Estimated price: {price:.2f} lakh</div>
          <div class="note">Model trained on {len(X):,} real Bengaluru
          listings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info(
        "👈 Enter the property details in the sidebar and hit Predict "
        "to see an estimate."
    )
