# 🏠 House Price Intelligence System

One end-to-end data-science project covering **all five units** of the
'Data Science using Python' (CAPCTD3) syllabus, built on Bengaluru
house price data. It takes a raw listings CSV through cleaning,
preprocessing, exploratory analysis and inference, visualization, and
a trained multiple linear regression model.

## Syllabus mapping

| Unit | What the project does | Where |
|---|---|---|
| I — Python, NumPy, Pandas | CSV loading, shape/dtypes/missing-value reports, NumPy indexing/slicing/reshaping/broadcasting on prices, `.loc`/`.iloc` selection, first cleaning pass | `src/house_price/loading.py`, `cleaning.py`, `parsing.py` |
| II — Preprocessing | Measurement levels, label + one-hot/dummy encoding, null handling, IQR & Z-score outlier filtering, mean/variance/std/covariance, Gaussian KS check, random + stratified sampling | `src/house_price/preprocessing.py` |
| III — Stats & inference | Descriptive stats, IQR, skewness, kurtosis, uni/bi/multivariate EDA, correlation, feature engineering, Bayes' theorem, Poisson/binomial distributions, CLT bootstrap CI, t-test, Z-test, one-way ANOVA | `src/house_price/eda_stats.py` |
| IV — Visualization | Matplotlib line/scatter/bar/hist/box plots with subplots; Seaborn distribution, relational, categorical plots, heatmap, pair plot, FacetGrid | `src/house_price/visualization.py` |
| V — Regression | Train/validation/test split, simple + multiple linear regression, residual plot, VIF multicollinearity check (via scikit-learn OLS), MSE & R², plus an optional KMeans market-segmentation touch | `src/house_price/modeling.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(A pre-built environment with the same pinned versions also works.)

## Run the pipeline

On the bundled sample (no Kaggle account needed):

```bash
python run_pipeline.py --sample
```

On the real Kaggle CSV (free Kaggle account; search "Bengaluru House
Price Data"; the file should appear as `Bengaluru_House_Data.csv`
after unzipping):

```bash
python run_pipeline.py data/Bengaluru_House_Data.csv --out-dir outputs
```

Outputs (all under `outputs/`):

- `phase1_clean.csv`, `phase2_features.csv`, `phase3_eda.csv`
- `matplotlib_dashboard.png`, `seaborn_dashboard.png`, `pairplot.png`,
  `facetgrid.png`, `residual_plot.png`
- `metrics.json` — descriptive stats, hypothesis-test results, model
  metrics (MSE, R²) and every skipped analysis with its reason

## Optional Streamlit dashboard

```bash
streamlit run app.py
```

Shows three key charts and a price-prediction form, using the same
cleaning/preprocessing/modelling code as the pipeline. The demo model
is trained on the bundled sample, so predictions are illustrative only.

## Run the tests

```bash
python -m pytest tests/ -q
```

Lint:

```bash
python -m ruff check src/ tests/ run_pipeline.py
```

## Parsing gotchas handled

The `total_sqft` column mixes formats, and the original plan sketch
mishandled units. This project uses the corrected conversions:

- `"1133 - 1384"` (range) → midpoint, 1258.5
- `"34.46Sq. Meter"` → 34.46 × 10.7639 ≈ 370.9 sqft (not 34.46)
- `"1200Sq. Yards"` → 1200 × 9 = 10800 sqft
- Unparseable values → NaN, counted, then dropped (never silently
  mis-converted)

## Honest limitations

- **Sample size.** The bundled sample has only 6 rows (5 after
  cleaning, 4 after outlier filtering). On it, the t-test and ANOVA
  are skipped (groups too small), VIF is skipped (fewer training rows
  than features), KMeans is skipped, and R² is `null` because a
  single test point cannot define it. All of these are recorded in
  `metrics.json` with reasons; none are faked. They run properly on
  the full ~13,000-row Kaggle dataset.
- **Real Kaggle CSV not verified here.** No Kaggle credentials were
  available during development, so the pipeline was verified
  end-to-end on the sample only. The code accepts any CSV with the
  same 9 columns.
- **No database, auth, or API server.** The data is a local CSV and
  there are no users or external services, so those components do
  not apply to this project and were deliberately not invented.
- **No model is persisted.** The pipeline trains fresh on each run
  and reports metrics; the optional Streamlit app (`app.py`) does the
  same at startup.
