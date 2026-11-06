# Kerala Air Quality Pattern Miner

**Multi-Source Data Integration and Pattern Discovery in Air Quality and Weather Data**

## 1. Project overview

The Kerala Air Quality Pattern Miner is a complete, runnable academic data-mining
project built around a professional Streamlit dashboard. It integrates independent
air-quality and weather observations for Kerala, cleans and preprocesses them,
and discovers patterns with:

* **Exploratory Data Analysis (EDA)** — interactive Plotly charts
* **K-Means clustering** — scikit-learn (similarity patterns, not rankings)
* **Apriori frequent-itemset and association-rule mining** — mlxtend (co-occurrence, not causation)

The application supports CSV upload (always available, no credentials needed) and
optional automated ingestion from OpenAQ (air quality) and NASA POWER (weather),
both guarded by API-key/network availability.

No synthetic data, fabricated metrics, hardcoded results, or decorative charts are
presented as analysis. Every chart, metric, cluster and rule is computed from the
data you load.

## 2. Objectives

* Integrate at least two independent datasets (air quality + weather) with
  documented join logic on date **and** location.
* Validate schemas, clean/preprocess transparently, and audit every decision.
* Run K-Means clustering and Apriori association-rule mining on real data.
* Export cleaned datasets, integration/preprocessing audits, cluster assignments
  and association rules as CSV.
* Document data sources, methodology, and limitations honestly.

## 3. Features

| Area | Capability |
|------|------------|
| Data import | Upload air-quality + weather CSVs; column mapping for differing schemas; schema validation with clear error messages |
| Data quality | Missing-value profiling (missing ≠ measured zero), duplicates, invalid dates/values, outlier flags, coverage by date/location |
| Integration | Join on date + location (never date alone); nearest-neighbour haversine colocation or explicit station→grid mapping; duplicate-key detection; merge-cardinality validation; matched/unmatched counts |
| Preprocessing | Timezone-aware date parsing (Asia/Kolkata), plausibility checks, outlier flagging (no silent deletion), feature engineering (month/weekday/season/monsoon), scaling, documented discretisation |
| EDA | Distributions, AQI, time series, location comparisons, monthly/seasonal profiles, missing-value bars, correlation heatmaps, AQ-vs-weather scatter |
| K-Means | Feature selection, k-range comparison, inertia + valid silhouette, cluster sizes/profiles, downloadable assignments |
| Apriori | Boolean transaction matrix, min support/confidence, itemsets + rules table, downloadable results, graceful no-rule handling |
| Exports | CSV downloads for all results + a Markdown methodology report |
| Sources | `DATA_SOURCES.md` documents verified vs unresolved source facts |

## 4. Architecture

The codebase separates ingestion, preprocessing, mining, reporting and UI logic:

```
kerala_air_quality_pattern_miner/
├── app.py                          # Streamlit entry point + session state
├── pages/                          # one page per dashboard section
│   ├── 01_overview.py
│   ├── 02_data_import.py
│   ├── 03_data_quality.py
│   ├── 04_integration.py
│   ├── 05_eda.py
│   ├── 06_kmeans.py
│   ├── 07_apriori.py
│   └── 08_exports.py
├── src/airquality/                 # core library (pure, testable)
│   ├── config.py                   # paths, constants, timezone/season policy
│   ├── ingestion.py                # CSV upload + optional OpenAQ / NASA POWER fetchers
│   ├── source_audit.py             # source provenance trail
│   ├── validation.py               # schema validation, data dictionary, quality profiling
│   ├── preprocessing.py            # cleaning pipeline + audits (no input mutation)
│   ├── integration.py              # multi-source integration + haversine colocation
│   ├── exploration.py              # Plotly EDA figures
│   ├── clustering.py               # scikit-learn K-Means
│   ├── association_rules.py        # mlxtend Apriori + rules
│   └── reporting.py                # exports + methodology report
├── data/raw/                       # user-uploaded raw CSVs (gitignored)
├── data/processed/                 # cleaned/integrated outputs (gitignored)
├── reports/figures/, reports/metrics/  # generated artefacts (gitignored)
├── tests/                          # pytest suite (35 tests)
├── docs/
├── DATA_SOURCES.md                 # verified source facts vs open questions
├── README.md
├── requirements.txt                # pinned dependencies
├── .env.example                    # copy to .env for secrets
└── .gitignore
```

## 5. Technology stack

* Python 3.11
* pandas>=2.0.0, NumPy>=2.0.0 — data structures
* scikit-learn>=1.3.0 — K-Means, scaling, silhouette
* mlxtend>=0.22.0 — Apriori / association rules
* Streamlit>=1.35.0 — dashboard
* Plotly>=5.15.0 — interactive charts
* pytest>=7.0.0 — automated tests

## 6. Prerequisites

* Windows 10/11 (or Linux/macOS), Python 3.10+
* Internet access only needed for optional remote fetchers; CSV upload works offline

## 7. Setup (Windows PowerShell)

```powershell
# 1. Navigate to the project folder
cd D:\Projects\DM_projectr\kerala_air_quality_pattern_miner

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure API keys (optional; CSV upload works without them)
copy .env.example .env
# .env already contains your OpenAQ API key. Never commit .env.

# 5. Run the application — datasets load automatically from data/raw/
streamlit run app.py
# Open http://localhost:8501 in your browser
```

### Auto-loaded datasets

On startup, the app automatically loads the following real datasets from `data/raw/` — no manual upload required:

| File | Rows | Coverage |
|------|------|----------|
| `cpcb_dly_aq_kerala-2008.csv` | Air quality | Kerala, 2008 |
| `cpcb_dly_aq_kerala-2009.csv` | Air quality | Kerala, 2009 |
| `cpcb_dly_aq_kerala-2010.csv` | Air quality | Kerala, 2010 |
| `cpcb_dly_aq_kerala-2011.csv` | Air quality | Kerala, 2011 |
| `cpcb_dly_aq_kerala-2015.csv` | Air quality | Kerala, 2015 |
| `POWER_Point_Daily_..._LST.csv` | 2,557 | Weather, Kerala/Kochi, 2019–2025 |

> **Note:** CPCB air-quality data covers 2008–2015; NASA POWER weather covers 2019–2025. They do not overlap in time, so the Integration page will show 0 matched rows for a date+location join. EDA, K-Means, and Apriori work on each dataset independently.

## 8. Setup (Linux / macOS)

```bash
cd kerala_air_quality_pattern_miner
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env   # edit .env if you have an OpenAQ API key
streamlit run app.py
```

## 9. API-key configuration

* `OPENAQ_API_KEY` — OpenAQ API v3 requires a key (send as `X-API-Key` header). Your key is already configured in `.env`. Get one at https://openaq.org/. Never commit `.env`.
  * **Verified:** Key works (HTTP 200 response), but **no Kerala stations** found across 4,000+ India locations queried. OpenAQ is not useful for Kerala data.
* `NASA_POWER_API_KEY` — optional registered token; the public NASA POWER API does not require one.

Read `DATA_SOURCES.md` for verified coverage and limitations of each source.

## 10. Obtaining and uploading data

Automated retrieval depends on source availability and credentials (see DATA_SOURCES.md).
The reliable path is **CSV upload**:

### Air-quality CSV (example columns)

| station | datetime | latitude | longitude | pm25 | no2 | so2 | o3 | aqi |
|---------|----------|----------|-----------|------|-----|-----|----|-----|
| Eloor | 2024-06-01 10:00 | 10.0 | 76.3 | 42.5 | 18.0 | 5.2 | 30.1 | 68 |

### Weather CSV (example columns)

| grid | datetime | latitude | longitude | temperature | humidity |
|------|----------|----------|-----------|-------------|----------|
| G_Eloor | 2024-06-01 10:00 | 10.05 | 76.35 | 28.0 | 80 |

On the **Data Import** page, map your file's columns to `datetime` and `location`
(required) plus optional `latitude`/`longitude`, then validate and load.

## 11. Running the application

```powershell
streamlit run app.py
```

Navigate the sidebar: Overview → Data Import → Data Quality → Integration &
Preprocessing → Exploratory Analysis → K-Means Clustering → Apriori Rules →
Exports & Methodology.

## 12. Running tests

```powershell
pytest
# or with a summary
pytest -v
```

35 automated tests cover valid/invalid schemas, missing required columns, invalid
dates, missing vs measured zeros, duplicates, duplicate join keys, merge
cardinality, row-multiplication prevention, matched/unmatched counts, unit
conversion, input preservation, empty datasets, invalid clustering features/counts,
clustering reproducibility, Apriori Boolean-matrix validation, no-rule outcomes,
CSV exports, and malformed-upload error handling.

Synthetic fixtures are explicitly labelled test data and are never used as results.

## 13. Data integration (how it works)

1. Each uploaded CSV is mapped to canonical columns (`datetime`, `location`,
   optional `latitude`/`longitude`).
2. Datetimes are parsed and localized to `Asia/Kolkata` (IST, UTC+05:30).
3. Temporal resolution of both datasets is inspected; a common grain (daily) is
   used only when supported.
4. Observations are matched on **date + location**. A join on date alone is never
   performed when locations differ.
5. When coordinates exist, each air-quality station is colocated with the nearest
   weather grid using haversine distance (max distance configurable); the method
   and distance are recorded. An explicit station→weather-grid mapping can be
   supplied instead.
6. Duplicate join keys are detected **before** merging and aggregated (mean for
   numerics) to prevent accidental row multiplication; merge cardinality is
   validated.
7. The audit reports input rows, output rows, matched rows, unmatched rows,
   duplicate keys, invalid keys, date/location coverage, and row-multiplication
   status.

## 14. Preprocessing (how it works)

* Missing values are profiled but distinguished from measured zeros and are never
  silently replaced with zero.
* Invalid (e.g. negative) concentrations are flagged as invalid; outliers are
  flagged with the IQR rule but not automatically deleted.
* Dates are parsed with an explicit timezone policy; invalid dates are counted,
  not silently dropped (optionally dropped with a count).
* Feature engineering adds month, weekday, and a Kerala season/monsoon indicator
  (Winter Dec–Feb; Pre-monsoon Mar–May; Southwest monsoon Jun–Sep; Post-monsoon
  Oct–Nov).
* Transformations are fitted only on the data provided (no data leakage).
* Raw data are preserved; cleaned and integrated datasets are saved separately.
* Every decision is recorded in a downloadable preprocessing audit.

## 15. K-Means (how it works)

* scikit-learn `KMeans` with a fixed `random_state` and explicit `n_init`.
* User selects eligible numeric features; features are standardised
  (`StandardScaler`).
* Rows with missing selected features are excluded and counted (never imputed).
* A range of cluster counts is compared; inertia and silhouette are reported only
  when mathematically valid (`2 <= k <= n_samples - 1`).
* Cluster numbers are arbitrary identifiers — **not** pollution rankings.
* Clustering discovers similarity patterns; it does not establish causation.

## 16. Apriori (how it works)

* Numeric features are discretised with documented bin edges (quantile or
  uniform); categorical columns are used directly.
* A Boolean transaction matrix is built and validated.
* mlxtend `apriori` + `association_rules` compute support, confidence and lift.
* Rules are displayed in a sortable, filterable table and are **co-occurrence
  patterns, not proof of causation**.
* When no itemsets/rules meet the thresholds, a clear message is shown (nothing
  is fabricated).

## 17. Output files and export formats

All exports are CSV (or Markdown for the methodology report):

* `cleaned_air_quality.csv`, `cleaned_weather.csv`
* `integrated_data.csv`
* `integration_audit.csv`, `preprocessing_audit_*.csv`
* `cluster_assignments.csv`, `cluster_profiles.csv`
* `frequent_itemsets.csv`, `association_rules.csv`
* `methodology_report.md`, `source_audit_log.csv`

## 18. Known limitations

* **OpenAQ:** API key works but zero Kerala stations found (4,000+ India locations checked). Use CPCB CSV files instead.
* **NASA POWER:** Downloaded file verified working (2019–2025 daily weather at Kochi). No temporal overlap with CPCB data (2008–2015) — integration produces 0 matched rows unless overlapping-year weather data is provided.
* **KSPCB:** Dashboard and dated AQI PDF widget exist; no confirmed public bulk API.
* Results depend entirely on the data you load; charts derive from the current dataset only.
* Cluster numbers and association rules are descriptive, not causal or predictive.

## 19. Troubleshooting

| Problem | Likely cause / fix |
|---------|--------------------|
| `streamlit` not found | Activate the venv: `.\.venv\Scripts\Activate.ps1` |
| Upload rejected | File must be `.csv`, within size limits, contain mapped `datetime` + `location` |
| "Missing required column(s)" | Use Data Import column mapper to map `datetime` and `location` |
| No charts shown | Data auto-loads on startup; if missing, check `data/raw/` has CSVs |
| Integration produces 0 rows | CPCB (2008–2015) and POWER (2019–2025) don't overlap in time — expected |
| No Apriori rules | Lower `min_support` / `min_confidence` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| `use_container_width` / `width` errors | Fixed in app — uses `use_container_width=True` for dataframe/plotly |
