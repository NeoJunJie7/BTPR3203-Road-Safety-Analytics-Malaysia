# BTPR3203 – Road Safety Analytics Malaysia

An end-to-end data analytics and machine learning pipeline investigating road accident trends in Malaysia, built for the BTPR3203 (Python for Data Science) course. The project combines multiple government open datasets to answer three research questions on vehicle involvement, accident severity classification, and state-level population-normalized hotspots — using pandas/numpy for data wrangling and scikit-learn Linear Regression for trend forecasting.

## Motivation

Road safety remains a pressing national issue in Malaysia. According to a report tabled in the Dewan Rakyat, **273,668 road accidents were reported between January and April 2026 alone**, reflecting a worrying upward trend ([The Star, 9 July 2026](https://www.thestar.com.my/news/nation/2026/07/09/worrying-upward-trend-in-road-accidents-with-273668-reported-from-january-to-april-dewan-rakyat-told)). This project uses historical open government data to understand the drivers behind this trend across vehicle types, accident classifications, and geography.

## Research Questions

1. **RQ1 – Vehicle Involvement:** Which vehicle types (motorcycle, car, lorry, etc.) are most frequently involved in reported road accidents in Malaysia, and how has this distribution changed over the years?
2. **RQ2 – Accident Classification:** What are the specific accident classification trends over the years in reported road accidents in Malaysia (e.g. traffic investigation cases, fatal accidents, deaths)?
3. **RQ3 – State Hotspots:** Which Malaysian states have the highest concentration of accidents and fatalities, and how does this compare once normalized by population?

## Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| Road accidents by type of vehicle | [data.gov.my archive](https://archive.data.gov.my/data/dataset/number-of-road-accidents-reported-by-type-of-vehicle-malaysia) | 2000–2019 |
| Deaths & injuries in road accidents | [data.gov.my archive](https://archive.data.gov.my/data/dataset/number-of-road-accidents-reported-by-type-of-vehicle-malaysia) | 2000–2021 |
| Statistik Kemalangan, Kemalangan Maut dan Kematian (state-level, by road type & classification) | [data.gov.my archive](https://archive.data.gov.my/data/ms_MY/dataset/statistik-laporan-kemalangan-jalan-raya-diterima-2010-2017) | 2016–2021 |
| State population estimates | [data.gov.my – Population by State](https://data.gov.my/data-catalogue/population_state) | 2016–2021 |

> **Note:** The state-level accident statistics are provided in both CSV and XLSX form from the same source; both are loaded, cross-validated against each other for consistency, and merged into a single deduplicated dataset (see [Data Cleaning](#data-cleaning--validation)).

## Project Structure

```
BTPR3203-Road-Safety-Analytics-Malaysia/
├── dataset/
│   ├── m-20210325040425_..._number-of-road-accidents-reported-by-type-of-vehicle-malaysia-2.csv
│   ├── m-20210329034857_..._2000-2021-number-of-deaths-and-injuries-in-road-accidents-repor.csv
│   ├── STATISTIK_KEMALANGAN__KEMALANGAN_MAUT_DAN_KEMATIAN.csv
│   ├── Statistik_Kemalangan__Kemalangan_Maut_Dan_Kematian.xlsx
│   └── state_population_2019.csv
├── outputs/                      # generated CSVs (created at runtime)
├── visualizations/               # generated figures (created at runtime)
├── Malaysian_Road_Accident_Data.py
└── README.md
```

> Place all raw files listed in the table above into a `dataset/` folder before running the script — the `outputs/` and `visualizations/` folders are created automatically on execution.

## Setup

**Requirements:** Python 3.9+

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Usage

```bash
python Malaysian_Road_Accident_Data.py
```

Running the script executes the full pipeline end-to-end: data loading/cleaning, exploratory processing for all three RQs, and Linear Regression forecasting for all three RQs — printing progress and summary tables to the console, and saving all processed data and figures to disk.

## Methodology

### Data Cleaning & Validation
- Column names and categorical values (state names, classification labels, vehicle types) are stripped of whitespace.
- Numeric columns with thousand-separator commas (e.g. `"1,029"`) are coerced to numeric.
- The state-level statistics are supplied as both a CSV and an XLSX from the same source; both are cleaned independently, then **cross-checked on overlapping records** (matched on Year, State, and Classification) to confirm they agree before being concatenated and deduplicated into one canonical dataset.
- Malaysian state names are standardized across datasets (e.g. `W.P. Kuala Lumpur` → `Kuala Lumpur`) so that accident and population data join correctly.

### RQ1 — Vehicle Type Involvement (2000–2019)
- Aggregates total involvement and percentage share by vehicle type.
- Pivots accidents by year × vehicle type to trace trends over two decades.
- **Forecasting:** A separate Linear Regression is fitted per vehicle type on the **2010–2019 stabilized reporting regime** (the earlier 2000s excluded due to a known reporting-methodology shift), then extrapolated to 2020–2026.

### RQ2 — Accident Classification Trends (2016–2021)
- Aggregates national deaths vs. injuries over time and computes a fatality ratio.
- Aggregates state-level accident classification totals (Traffic Investigation Cases, Fatal Accidents, Deaths) nationally by year, and computes a fatality rate (deaths per case opened).
- **Forecasting:** A Linear Regression per classification is trained on the **pre-pandemic 2016–2019 baseline** and forecast to 2022–2026, explicitly excluding the COVID-19-distorted 2020–2021 window from training (shown in visualizations for context only, shaded gray).

### RQ3 — State Hotspots Normalized by Population
- Merges state-level accident/fatality totals with state population data (filtered to `sex == both`, `age == overall`, `ethnicity == overall`) to compute **accidents and fatalities per 100,000 population**.
- Compares raw accident counts against population-normalized rates to reveal which states are genuine hotspots versus states that merely have larger populations.
- **Forecasting:** A Linear Regression per state, per target (`Accidents_per_100k`, `Fatalities_per_100k`) is trained on the **2016–2019 pre-pandemic baseline** and forecast to 2022–2026.

### Forecasting Design Notes
All three forecasting modules share a consistent, deliberate design:
- Models are trained only on a **stable baseline window**, explicitly excluding periods known to distort trends (pre-2010 reporting-regime shift for RQ1; the 2020–2021 pandemic dip for RQ2/RQ3).
- Forecast lines are **anchored at the model's fitted value** for the last baseline year (not the raw, possibly distorted actual value), so forecasts continue the underlying trend smoothly instead of kinking off an anomalous data point.
- Predictions are floored at zero (`np.maximum(0, preds)`) since negative accident counts are not meaningful.
- Model fit is evaluated in-sample on the baseline window using **R², RMSE, and MAE**.

## Outputs

Running the script produces the following in `outputs/`:

| File | Description |
|---|---|
| `processed_vehicle_accidents_by_year.csv` | Accidents by year × vehicle type (RQ1) |
| `feature_vehicle_growth_rates.csv` | Year-over-year % growth by vehicle type (RQ1) |
| `rq1_model_evaluation_metrics.csv` | R²/RMSE/MAE per vehicle type model (RQ1) |
| `rq1_future_vehicle_predictions.csv` | 2020–2026 forecast by vehicle type (RQ1) |
| `processed_deaths_and_injuries_trend.csv` | Deaths/injuries trend + fatality ratio (RQ2) |
| `processed_national_classification_2016_2021.csv` | National classification totals by year (RQ2) |
| `feature_rq2_classification_trend.csv` | Classification trend + fatality rate feature (RQ2) |
| `rq2_model_evaluation_metrics.csv` | R²/RMSE/MAE per classification model (RQ2) |
| `rq2_future_classification_predictions.csv` | 2022–2026 forecast by classification (RQ2) |
| `processed_state_accidents_normalized_2019.csv` | State accidents/fatalities, raw + per-100k (RQ3) |
| `feature_rq3_state_population_normalized.csv` | Full state-year panel with population-normalized rates (RQ3) |
| `rq3_model_evaluation_metrics.csv` | R²/RMSE/MAE per state/target model (RQ3) |
| `rq3_future_state_rate_predictions.csv` | 2022–2026 forecast by state and rate (RQ3) |

And in `visualizations/`:

| Figure | Description |
|---|---|
| `fig1_vehicle_trends.png` | Trends of accident involvement, top 5 vehicle types (2000–2019) |
| `fig2_deaths_vs_injuries.png` | Deaths vs. injuries dual-axis trend (2000–2019) |
| `fig3_traffic_cases_trend.png` | Total traffic investigation cases opened (2016–2021) |
| `fig4_state_accidents_raw_vs_normalized.png` | Raw vs. population-normalized accidents by state (2019) |
| `fig5_state_fatalities_normalized.png` | Fatality rate per 100k population by state (2019) |
| `fig6_rq1_ml_predictions.png` | RQ1 Linear Regression forecast, top 5 vehicle types |
| `fig7_rq2_ml_predictions.png` | RQ2 Linear Regression forecast by classification |
| `fig8_rq3_ml_predictions.png` | RQ3 Linear Regression forecast, top 5 states, both rates |

## Tech Stack

- **pandas / numpy** — data cleaning, merging, feature engineering
- **matplotlib / seaborn** — static visualizations
- **scikit-learn** — Linear Regression modeling and evaluation (R², RMSE, MAE)

## Limitations

- The vehicle-type and injury/death datasets run through 2019/2021 respectively, while the state-level classification dataset only covers 2016–2021; the three RQs therefore operate on different (though overlapping) time windows dictated by data availability.
- A known transposition artifact exists in the 2010 records for Perak and Pulau Pinang in the source state-level data; this is a documented limitation of the raw dataset rather than a processing error.
- Forecasts are simple linear extrapolations of past trends and do not account for policy interventions, enforcement changes, or other external factors — they should be read as trend continuations, not predictions of certainty.
- State population figures are only available up to 2021 in the datasets used, capping the population-normalization window accordingly.

## Course Context

This project was developed for **BTPR3203 (Python for Data Science)**, Southern University College, 2026B semester.
