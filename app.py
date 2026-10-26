"""Streamlit dashboard.

Loads the real Bengaluru dataset, runs the same cleaning,
preprocessing and modelling code as the pipeline, shows three key
charts, and offers a price-prediction form. The model is trained on
all cleaned, outlier-filtered real listings (see the README).

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from house_price import cleaning, eda_stats, loading, modeling, preprocessing

st.set_page_config(page_title="House Price Intelligence", layout="wide")
st.title("🏠 House Price Intelligence System")
st.write(
    "Bengaluru house prices, from raw CSV to a trained regression "
    "model — covering all five units of the Data Science using Python "
    "(CAPCTD3) syllabus."
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
col1, col2, col3 = st.columns(3)
col1.metric("Listings (cleaned)", len(eda))
col2.metric("Mean price (lakh)", f"{eda['price'].mean():.2f}")
col3.metric("Median price (lakh)", f"{eda['price'].median():.2f}")

st.subheader("Charts")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(eda["price"], bins=5)
axes[0].set_title("Price distribution")
axes[0].set_xlabel("Price (lakh)")
axes[1].scatter(eda["total_sqft"], eda["price"])
axes[1].set_title("Sqft vs price")
axes[1].set_xlabel("Total sqft")
axes[1].set_ylabel("Price (lakh)")
med = eda.groupby("bhk")["price"].median().sort_index()
axes[2].bar(med.index.astype(str), med.values)
axes[2].set_title("Median price by BHK")
axes[2].set_xlabel("BHK")
fig.tight_layout()
st.pyplot(fig)

st.subheader("Predict a price")
with st.form("predict"):
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
    st.success(f"Estimated price: **{price:.2f} lakh** "
               f"(model trained on {len(X)} real listings)")
