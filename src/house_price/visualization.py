"""Phase 4 visualization dashboard (Unit IV).

Matplotlib: line, scatter, bar, histogram and box plots, titles,
labels, legends, styles and subplots. Seaborn: distribution, relational
and categorical plots, heatmaps, pair plots and faceting.

All figures are saved as PNG files (Agg backend, no display needed).
Every plot guards against tiny inputs so the dashboard still renders
on the 5-row sample as well as on the full dataset.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from .eda_stats import correlation_matrix

sns.set_theme(style="whitegrid")


def _sample(df, n):
    return df.sample(min(n, len(df)), random_state=42) if len(df) else df


def matplotlib_dashboard(df, path):
    """2x3 Matplotlib dashboard: scatter, hist, box, bar, line, bar."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    price = df["price"]
    clip_price = price.clip(upper=500) if price.max() > 500 else price

    axes[0, 0].scatter(df["total_sqft"], price, alpha=0.3)
    axes[0, 0].set_title("Sqft vs Price")
    axes[0, 0].set_xlabel("Total sqft")
    axes[0, 0].set_ylabel("Price (lakh)")

    axes[0, 1].hist(clip_price, bins=min(60, max(2, len(df) // 2)))
    axes[0, 1].set_title("Price Distribution")
    axes[0, 1].set_xlabel("Price (lakh)")

    axes[0, 2].boxplot(clip_price.dropna())
    axes[0, 2].set_title("Price Boxplot")
    axes[0, 2].set_ylabel("Price (lakh)")

    top = df["location"].value_counts().head(10)
    axes[1, 0].bar(top.index.astype(str), top.values)
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].set_title("Top Localities by Listings")

    med_by_bhk = df.groupby("bhk")["price"].median().sort_index()
    axes[1, 1].plot(med_by_bhk.index, med_by_bhk.values, marker="o")
    axes[1, 1].set_title("Median Price by BHK")
    axes[1, 1].set_xlabel("BHK")
    axes[1, 1].set_ylabel("Median price (lakh)")

    # NOTE: area_type was one-hot encoded in Phase 2, so it no longer
    # exists as a column here; bath count is still available.
    med_by_bath = df.groupby("bath")["price"].median().sort_index()
    axes[1, 2].bar(med_by_bath.index.astype(str), med_by_bath.values)
    axes[1, 2].tick_params(axis="x", rotation=45)
    axes[1, 2].set_title("Median Price by Bath Count")

    fig.suptitle("Bengaluru Housing — Matplotlib Dashboard", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def seaborn_dashboard(df, path):
    """2x2 Seaborn dashboard: distribution, relational, categorical, heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["price_per_sqft"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of price/sqft")

    sns.regplot(x="total_sqft", y="price", data=_sample(df, 2000),
                scatter_kws={"alpha": 0.2}, ax=axes[0, 1])
    axes[0, 1].set_title("Relational: sqft → price")

    sns.boxplot(x="location_tier", y="price", data=df, ax=axes[1, 0])
    axes[1, 0].set_title("Categorical: price by location tier")

    corr = correlation_matrix(df)
    if not corr.empty:
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[1, 1])
        axes[1, 1].set_title("Correlation Heatmap")
    else:
        axes[1, 1].set_title("Correlation Heatmap (not enough varying columns)")

    fig.suptitle("Bengaluru Housing — Seaborn Dashboard", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def pairplot(df, path):
    """Seaborn pair plot over the core numeric features."""
    cols = [c for c in ["price", "total_sqft", "bath", "bhk", "location_tier"]
            if c in df.columns]
    grid = sns.pairplot(_sample(df, 500)[cols], diag_kind="hist")
    grid.fig.suptitle("Pair Plot — Core Features", y=1.02)
    grid.fig.savefig(path, dpi=120)
    plt.close("all")
    return path


def facetgrid(df, path):
    """FacetGrid of sqft vs price, faceted by premium-location flag."""
    if "is_premium" not in df.columns:
        return None
    grid = sns.FacetGrid(df, col="is_premium", height=4, aspect=1.2)
    grid.map(sns.scatterplot, "total_sqft", "price", alpha=0.3)
    grid.set_titles("Premium locality: {col_name}")
    grid.fig.savefig(path, dpi=120)
    plt.close("all")
    return path
