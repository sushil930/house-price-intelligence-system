"""Streamlit dashboard.

Loads the real Bengaluru dataset, runs the same cleaning,
preprocessing and modelling code as the pipeline, shows key
metrics and charts, and offers a sidebar price-prediction form.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from house_price import cleaning, eda_stats, loading, modeling, preprocessing

st.set_page_config(
    page_title="House price intelligence",
    page_icon=":material/analytics:",
    layout="wide",
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

# Header
st.title("House price intelligence", icon=":material/home_work:")
st.caption(
    "Bengaluru listings — from raw CSV to a trained regression model. "
    "Use the sidebar form to estimate a price."
)

# KPI row
with st.container(horizontal=True):
    st.metric("Listings (cleaned)", f"{len(eda):,}", border=True)
    st.metric("Mean price", f"{eda['price'].mean():.1f} lakh", border=True)
    st.metric("Median price", f"{eda['price'].median():.1f} lakh", border=True)
    st.metric(
        "Median price/sqft",
        f"₹{eda['price_per_sqft'].median():,.0f}",
        border=True,
    )

# Charts row
chart_height = 230
col1, col2, col3 = st.columns(3, border=True)
with col1:
    st.subheader("Price distribution")
    hist = (
        alt.Chart(eda)
        .mark_bar()
        .encode(
            alt.X("price:Q", bin=alt.Bin(maxbins=25), title="Price (lakh)"),
            alt.Y("count():Q", title="Listings"),
        )
        .properties(height=chart_height)
    )
    st.altair_chart(hist, height=chart_height, alt="Histogram of listing prices")
with col2:
    st.subheader("Sqft vs price")
    st.scatter_chart(
        eda,
        x="total_sqft",
        y="price",
        x_label="Total sqft",
        y_label="Price (lakh)",
        height=chart_height,
        alt="Scatter plot of total sqft against price",
    )
with col3:
    st.subheader("Median price by BHK")
    med = eda.groupby("bhk")["price"].median().reset_index()
    med["bhk"] = med["bhk"].astype(str)
    st.bar_chart(
        med,
        x="bhk",
        y="price",
        x_label="BHK",
        y_label="Median price (lakh)",
        height=chart_height,
        alt="Bar chart of median price by BHK",
    )

# Sidebar prediction form
with st.sidebar:
    st.header("Predict a price", icon=":material/query_stats:")
    st.caption("Enter the property details and hit Predict.")
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
        submitted = st.form_submit_button(
            "Predict", type="primary", width="stretch"
        )

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
        X_new = pd.DataFrame([[row[c] for c in features]], columns=features)
        price = float(model.predict(X_new)[0])
        with st.container(border=True):
            st.metric("Estimated price", f"{price:,.2f} lakh")
            st.caption(f"Trained on {len(X):,} cleaned Bengaluru listings")
    else:
        st.caption("Your estimate will appear here after you predict.")
