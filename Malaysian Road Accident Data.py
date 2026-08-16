import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set overall plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})

def setup_directories():
    #Ensure output folders exist for processed files and visualizations.
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    print("[INFO] Output directories initialized.")

def load_and_clean_data():
    #Load raw datasets, clean formatting, and convert data types.
    print("[INFO] Loading datasets...")
    
    # 1. Load Vehicle Dataset
    df_veh = pd.read_csv('dataset/m-20210325040425_202103250404250_number-of-road-accidents-reported-by-type-of-vehicle-malaysia-2.csv')
    df_veh['Type of Vehicle'] = df_veh['Type of Vehicle'].str.strip()
    
    # 2. Load Injury/Casualty Dataset
    df_inj = pd.read_csv('dataset/m-20210329034857_202307060208460_2000-2021-number-of-deaths-and-injuries-in-road-accidents-repor.csv')
    df_inj['Type of road user'] = df_inj['Type of road user'].str.strip()
    df_inj['Condition'] = df_inj['Condition'].str.strip()
    
    # 3. Load State Statistics Dataset
    df_stat1 = pd.read_csv('dataset/STATISTIK KEMALANGAN, KEMALANGAN MAUT DAN KEMATIAN.csv')
    df_stat1 = df_stat1.dropna(how='all', axis=1)

    df_stat2 = pd.read_excel('dataset/Statistik Kemalangan, Kemalangan Maut Dan Kematian.xlsx', header=1)

    key_cols = ['Tahun', 'Negeri', 'Klasifikasi Kemalangan']
    num_cols = ['Lebuh Raya', 'Jalan Persekutuan', 'Jalan Negeri', 'Jalan Bandaran', 'Lain-lain Jalan', 'Jumlah']

    # Clean state datasets
    def clean_dataframe(df):
        df = df.dropna(subset=['Tahun']).copy()
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Clean and cast key columns
        df['Tahun'] = df['Tahun'].astype(int)
        df['Negeri'] = df['Negeri'].astype(str).str.strip()
        df['Klasifikasi Kemalangan'] = df['Klasifikasi Kemalangan'].astype(str).str.strip()
        # Clean numeric columns (remove commas like '1,029' -> 1029)
        for col in num_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
        return df

    df_stat1_clean = clean_dataframe(df_stat1)
    df_stat2_clean = clean_dataframe(df_stat2)

    # Compare overlapping records
    overlap = pd.merge(df_stat1_clean, df_stat2_clean, on=key_cols, suffixes=('_excel', '_csv'))
    diff_found = False
    for col in num_cols:
        diff_rows = overlap[overlap[f'{col}_excel'] != overlap[f'{col}_csv']]
        if not diff_rows.empty:
            diff_found = True
            print(f"WARNING: Differences found in column '{col}': {len(diff_rows)} rows mismatch.")
            print(diff_rows[key_cols + [f'{col}_excel', f'{col}_csv']].head())
    if not diff_found:
        print("All 126 overlapping rows match perfectly across all columns.")

    # Concatenate and remove duplicates based on the primary keys
    df_combined = pd.concat([df_stat1_clean, df_stat2_clean], ignore_index=True)
    df_stat = df_combined.drop_duplicates(subset=key_cols, keep='last')

    # Sort chronologically and by location
    df_stat = df_stat.sort_values(by=['Tahun', 'Negeri', 'Klasifikasi Kemalangan']).reset_index(drop=True)
    
    # Clean commas and convert object columns to numeric in State dataset
    num_cols = ['Lebuh Raya', 'Jalan Persekutuan', 'Jalan Negeri', 'Jalan Bandaran', 'Lain-lain Jalan', 'Jumlah']
    for col in num_cols:
        if col in df_stat.columns:
            df_stat[col] = df_stat[col].astype(str).str.replace(',', '').str.strip()
            df_stat[col] = pd.to_numeric(df_stat[col], errors='coerce')
            
    df_stat = df_stat.drop(columns=['Unnamed: 9'], errors='ignore')
    
    print("[INFO] Data loading and cleaning complete.")
    return df_veh, df_inj, df_stat

def process_rq1(df_veh):
    #RQ1: Vehicle involvement analysis and longitudinal distribution.
    
    print("\n--- Processing RQ1: Vehicle Types Analysis ---")
    
    # Total involvement summary across all years
    veh_summary = df_veh.groupby('Type of Vehicle')['Value'].sum().sort_values(ascending=False).reset_index()
    veh_summary['Percentage'] = (veh_summary['Value'] / veh_summary['Value'].sum()) * 100
    
    print("Vehicle Involvement Summary (2000-2019):")
    print(veh_summary.to_string(index=False))
    
    # Pivot dataset by year and vehicle type
    veh_pivot = df_veh.pivot(index='Year', columns='Type of Vehicle', values='Value')
    
    # Save output deliverable
    veh_pivot.to_csv('outputs/processed_vehicle_accidents_by_year.csv')
    print("[OUTPUT SAVED] outputs/processed_vehicle_accidents_by_year.csv")
    
    # Plotting Figure 1
    plt.figure(figsize=(10, 6))
    top_vehicles = veh_pivot.sum().sort_values(ascending=False).index[:5]
    for col in top_vehicles:
        plt.plot(veh_pivot.index, veh_pivot[col], marker='o', linewidth=2, label=col)
        
    plt.title('Figure 1: Trends of Road Accident Involvement by Top 5 Vehicle Types (2000–2019)', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Vehicles Involved')
    plt.legend(title='Vehicle Type')
    plt.tight_layout()
    plt.savefig('visualizations/fig1_vehicle_trends.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig1_vehicle_trends.png")

def process_rq2(df_inj, df_stat):
    #RQ2: Accident classification trends (Deaths vs. Injuries & State Classifications).

    print("\n--- Processing RQ2: Accident Severity & Classifications ---")
    
    # Aggregate Deaths vs. Injuries
    inj_trend = df_inj.groupby(['Year', 'Condition'])['Value'].sum().unstack()
    
    # Feature Engineering: Fatality Ratio
    inj_trend['Total_Casualties'] = inj_trend['Deaths'] + inj_trend['Injuries']
    inj_trend['Fatality_Ratio_Pct'] = (inj_trend['Deaths'] / inj_trend['Total_Casualties']) * 100
    
    print("Casualty Severity Summary (Sample Years):")
    print(inj_trend.head())
    
    # Save casualty output deliverable
    inj_trend.to_csv('outputs/processed_deaths_and_injuries_trend.csv')
    print("[OUTPUT SAVED] outputs/processed_deaths_and_injuries_trend.csv")
    
    # State classification national aggregation (2016-2021)
    stat_trend = df_stat.groupby(['Tahun', 'Klasifikasi Kemalangan'])['Jumlah'].sum().unstack()
    stat_trend.to_csv('outputs/processed_national_classification_2016_2021.csv')
    print("[OUTPUT SAVED] outputs/processed_national_classification_2016_2021.csv")
    
    # Plotting Figure 2: Deaths vs Injuries dual axis
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Deaths', color=color, fontweight='bold')
    line1 = ax1.plot(inj_trend.index, inj_trend['Deaths'], color=color, marker='o', linewidth=2, label='Deaths')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Number of Injuries', color=color, fontweight='bold')
    line2 = ax2.plot(inj_trend.index, inj_trend['Injuries'], color=color, marker='s', linewidth=2, linestyle='--', label='Injuries')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Figure 2: Road Accident Severity Trend in Malaysia (2000–2019): Deaths vs Injuries', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/fig2_deaths_vs_injuries.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig2_deaths_vs_injuries.png")
    
    # Plotting Figure 3: Traffic Investigation Cases Trend
    plt.figure(figsize=(9, 5))
    plt.plot(stat_trend.index, stat_trend['Kes Siasatan Trafik (KST) Kemalangan Jalan Raya Dibuka'], marker='o', linewidth=2, color='darkblue')
    plt.title('Figure 3: Total Reported Traffic Investigation Cases in Malaysia (2016–2021)', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Total Cases Opened')
    plt.xticks(stat_trend.index)
    plt.tight_layout()
    plt.savefig('visualizations/fig3_traffic_cases_trend.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig3_traffic_cases_trend.png")

def process_rq3(df_stat):
    #RQ3: State concentration analysis normalized by population metrics using state_population_2019.csv.
    
    print("\n--- Processing RQ3: Population-Normalized State Hotspots ---")
    
    # 1. Filter 2019 accident baseline data
    df_2019 = df_stat[df_stat['Tahun'] == 2019].copy()
    state_2019 = df_2019.pivot(index='Negeri', columns='Klasifikasi Kemalangan', values='Jumlah').reset_index()
    state_2019.columns.name = None
    
    state_2019 = state_2019.rename(columns={
        'Kes Siasatan Trafik (KST) Kemalangan Jalan Raya Dibuka': 'Total_Accidents',
        'Kemalangan Maut': 'Fatal_Accidents',
        'Kematian': 'Fatalities'
    })
    
    # 2. Load and filter external population dataset
    df_pop = pd.read_csv('dataset/state_population_2019.csv')
    df_pop['date'] = pd.to_datetime(df_pop['date'])
    
    df_pop_2019 = df_pop[
        (df_pop['date'].dt.year == 2019) & 
        (df_pop['sex'] == 'both') & 
        (df_pop['age'] == 'overall') & 
        (df_pop['ethnicity'] == 'overall')
    ].copy()
    
    # Standardize state names & convert population scale (thousands -> actual count)
    df_pop_2019['state'] = df_pop_2019['state'].replace({'W.P. Kuala Lumpur': 'Kuala Lumpur'})
    df_pop_2019['Population'] = df_pop_2019['population'] * 1000
    
    # 3. Merge datasets using left_on='Negeri' and right_on='state'
    state_2019 = state_2019.merge(
        df_pop_2019[['state', 'Population']], 
        left_on='Negeri', 
        right_on='state', 
        how='left'
    ).drop(columns=['state'])
    
    # 4. Feature Engineering: Normalization per 100,000 population
    state_2019['Accidents_per_100k'] = (state_2019['Total_Accidents'] / state_2019['Population']) * 100000
    state_2019['Fatalities_per_100k'] = (state_2019['Fatalities'] / state_2019['Population']) * 100000
    
    # Save state normalized dataset
    state_2019.to_csv('outputs/processed_state_accidents_normalized_2019.csv', index=False)
    print("[OUTPUT SAVED] outputs/processed_state_accidents_normalized_2019.csv")
    
    # Plotting Figure 4: Raw vs Normalized Accidents
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    state_raw_sorted = state_2019.sort_values(by='Total_Accidents', ascending=False)
    sns.barplot(data=state_raw_sorted, x='Total_Accidents', y='Negeri', hue='Negeri', legend=False, palette='viridis')
    plt.title('Absolute Road Accidents by State (2019)', fontweight='bold')
    plt.xlabel('Total Accidents Reported')
    plt.ylabel('State')
    
    plt.subplot(1, 2, 2)
    state_norm_sorted = state_2019.sort_values(by='Accidents_per_100k', ascending=False)
    sns.barplot(data=state_norm_sorted, x='Accidents_per_100k', y='Negeri', hue='Negeri', legend=False, palette='magma')
    plt.title('Normalized Accidents per 100,000 Pop (2019)', fontweight='bold')
    plt.xlabel('Accidents per 100k Population')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('visualizations/fig4_state_accidents_raw_vs_normalized.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig4_state_accidents_raw_vs_normalized.png")
    
    # Plotting Figure 5: Normalized Fatality Rates
    plt.figure(figsize=(10, 5))
    state_fat_sorted = state_2019.sort_values(by='Fatalities_per_100k', ascending=False)
    sns.barplot(data=state_fat_sorted, x='Fatalities_per_100k', y='Negeri', hue='Negeri', legend=False, palette='Reds_r')
    plt.title('Figure 5: Road Fatality Rate per 100,000 Population by Malaysian State (2019)', fontweight='bold')
    plt.xlabel('Fatalities per 100,000 Population')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig('visualizations/fig5_state_fatalities_normalized.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig5_state_fatalities_normalized.png")

def ml_predict_rq1(df_veh):
    #RQ1 Machine Learning Forecast:
    #Trained on the modern stabilized period (2010-2019) to prevent distortion
    #from pre-2005 administrative reporting changes.

    print("\n" + "="*60)
    print("  RQ1 MACHINE LEARNING: REALISTIC VEHICLE FORECASTING")
    print("="*60)
    
    veh_pivot = df_veh.pivot(index='Year', columns='Type of Vehicle', values='Value')
    
    # Feature Engineering: Historical Growth
    veh_growth = veh_pivot.pct_change() * 100
    veh_growth.to_csv('outputs/feature_vehicle_growth_rates.csv')
    
    # Target Years
    future_years = np.array([2024, 2025, 2026]).reshape(-1, 1)
    
    # Train on Modern Baseline (2010-2019)
    train_pivot = veh_pivot.loc[2010:2019]
    X_train = train_pivot.index.values.reshape(-1, 1)
    
    metrics_list = []
    future_preds = {'Year': [2024, 2025, 2026]}
    
    for vehicle in veh_pivot.columns:
        y_train = train_pivot[vehicle].values
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # In-sample Evaluation
        y_pred = model.predict(X_train)
        metrics_list.append({
            'Vehicle_Type': vehicle,
            'R2_Score': round(r2_score(y_train, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_train, y_pred)), 2),
            'MAE': round(mean_absolute_error(y_train, y_pred), 2)
        })
        
        # Future Predictions
        preds = model.predict(future_years)
        future_preds[vehicle] = np.maximum(0, preds.round(0))
        
    df_metrics = pd.DataFrame(metrics_list)
    df_predictions = pd.DataFrame(future_preds)
    
    df_metrics.to_csv('outputs/rq1_model_evaluation_metrics.csv', index=False)
    df_predictions.to_csv('outputs/rq1_future_vehicle_predictions.csv', index=False)
    
    print("\n--- Model Success Metrics (2010-2019 Regime) ---")
    print(df_metrics.to_string(index=False))
    
    print("\n--- Corrected Future Predictions (2024-2026) ---")
    print(df_predictions.to_string(index=False))
    
    # Visualization: Top 5 Vehicles with Continuous Trend
    top_5 = ['Motorcycle', 'Motorcar', 'Pedestrian', 'Bicycle', 'Others']
    
    plt.figure(figsize=(10, 5))
    for v in top_5:
        # Plot historical line
        plt.plot(veh_pivot.index, veh_pivot[v], marker='o', label=f'{v} (Actual)', alpha=0.7)
        # Connect last actual point (2019) to future predictions
        plot_years = np.append([2019], future_years.flatten())
        plot_vals = np.append([veh_pivot.loc[2019, v]], df_predictions[v].values)
        plt.plot(plot_years, plot_vals, linestyle='--', marker='s', label=f'{v} (Forecast)')
        
    plt.title('Figure 6: Corrected ML Linear Regression Forecast for Top Vehicle Types (2000–2026)', fontsize=11, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Reported Accidents / Involvements')
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/fig6_rq1_ml_predictions.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig6_rq1_ml_predictions.png")
    
    return df_metrics, df_predictions

def ml_predict_rq2(df_stat):
    #RQ2 Machine Learning Forecast:
    #Which specific accident classification trends over the years in
    #reported road accidents in Malaysia?

    #Forecasts the national-level yearly count (Jumlah) for each of the
    #three accident classifications (KST cases opened, Fatal accidents,
    #Deaths) using Linear Regression, trained on the full 2016-2021
    #national trend (joined from the 2016-2019 and 2017-2021 sources).

    print("\n" + "=" * 60)
    print("  RQ2 MACHINE LEARNING: ACCIDENT CLASSIFICATION FORECASTING")
    print("=" * 60)

    # --- Step 1: National aggregation per Year x Classification ---
    class_trend = df_stat.groupby(['Tahun', 'Klasifikasi Kemalangan'])['Jumlah'].sum().unstack()

    kst_col = 'Kes Siasatan Trafik (KST) Kemalangan Jalan Raya Dibuka'
    fatal_col = 'Kemalangan Maut'
    death_col = 'Kematian'
    classifications = [kst_col, fatal_col, death_col]

    # --- Step 2: Feature Engineering - Fatality Rate trend ---
    # Deaths per KST case opened, expressed as a percentage.
    class_trend['Fatality_Rate_Pct'] = (class_trend[death_col] / class_trend[kst_col]) * 100
    class_trend.to_csv('outputs/feature_rq2_classification_trend.csv')

    print("\nNational Classification Trend (with Fatality Rate feature):")
    print(class_trend.round(2).to_string())

    # --- Step 3 & 4: Forecast target - Linear Regression per classification ---
    train = class_trend.dropna(subset=classifications)
    X_train = train.index.values.reshape(-1, 1)

    future_years = np.array([2022, 2023, 2024]).reshape(-1, 1)

    metrics_list = []
    future_preds = {'Year': future_years.flatten()}

    for cls in classifications:
        y_train = train[cls].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        # In-sample evaluation
        y_pred = model.predict(X_train)

        # --- Step 5: Success metrics ---
        metrics_list.append({
            'Classification': cls,
            'RMSE': round(np.sqrt(mean_squared_error(y_train, y_pred)), 2),
            'MAE': round(mean_absolute_error(y_train, y_pred), 2),
            'R2_Score': round(r2_score(y_train, y_pred), 4)
        })

        preds = model.predict(future_years)
        future_preds[cls] = np.maximum(0, preds.round(0))

    df_metrics = pd.DataFrame(metrics_list)
    df_predictions = pd.DataFrame(future_preds)

    df_metrics.to_csv('outputs/rq2_model_evaluation_metrics.csv', index=False)
    df_predictions.to_csv('outputs/rq2_future_classification_predictions.csv', index=False)

    print("\n--- Model Success Metrics (trained on 2016-2021 national trend) ---")
    print(df_metrics.to_string(index=False))

    print("\n--- Forecasted Classification Counts (2022-2024) ---")
    print(df_predictions.to_string(index=False))

    # --- Visualization: Actual vs Forecast per classification ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Actual': 'darkblue', 'Forecast': 'tab:red'}

    for ax, cls in zip(axes, classifications):
        ax.plot(train.index, train[cls], marker='o', linewidth=2,
                color=colors['Actual'], label='Actual')

        plot_years = np.append([train.index.max()], future_years.flatten())
        plot_vals = np.append([train[cls].iloc[-1]], df_predictions[cls].values)
        ax.plot(plot_years, plot_vals, linestyle='--', marker='s', linewidth=2,
                color=colors['Forecast'], label='Forecast')

        ax.set_title(cls, fontsize=10, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.legend()

    plt.suptitle('Figure 7: RQ2 Linear Regression Forecast by Accident Classification (2016-2024)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/fig7_rq2_ml_predictions.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig7_rq2_ml_predictions.png")

    return df_metrics, df_predictions

def ml_predict_rq3(df_stat, train_years=(2016, 2019), forecast_years=range(2022, 2027)):
    # RQ3 Machine Learning Forecast:
    # Which Malaysian states have the highest concentration of accidents and
    # fatalities, and how does this compare once normalized by population?

    # Joins the yearly national accident classification data with yearly state
    # population data on ['Tahun', 'Negeri'], engineers Accidents_per_100k and
    # Fatalities_per_100k, then fits one Linear Regression per state to
    # forecast each rate forward.

    print("\n" + "=" * 60)
    print("  RQ3 MACHINE LEARNING: STATE ACCIDENT/FATALITY RATE FORECASTING")
    print("=" * 60)
    print(f"  Historical data available : 2016-2021 (full range, plotted for context)")
    print(f"  Model trained on          : {train_years[0]}-{train_years[1]} (pre-pandemic baseline)")
    print(f"  Forecast horizon          : {min(forecast_years)}-{max(forecast_years)}")
    print("=" * 60)

    # --- Step 1: Join accident data with population data on Tahun + Negeri ---
    state_year = df_stat.pivot_table(
        index=['Tahun', 'Negeri'],
        columns='Klasifikasi Kemalangan',
        values='Jumlah'
    ).reset_index()
    state_year.columns.name = None
    state_year = state_year.rename(columns={
        'Kes Siasatan Trafik (KST) Kemalangan Jalan Raya Dibuka': 'Total_Accidents',
        'Kemalangan Maut': 'Fatal_Accidents',
        'Kematian': 'Fatalities'
    })

    df_pop = pd.read_csv('dataset/state_population_2019.csv')
    df_pop['date'] = pd.to_datetime(df_pop['date'])
    df_pop_yearly = df_pop[
        (df_pop['sex'] == 'both') &
        (df_pop['age'] == 'overall') &
        (df_pop['ethnicity'] == 'overall')
    ].copy()
    df_pop_yearly['Tahun'] = df_pop_yearly['date'].dt.year

    # Standardize state name spelling to match accident dataset's 'Negeri' labels
    df_pop_yearly['Negeri'] = df_pop_yearly['state'].replace({
        'W.P. Kuala Lumpur': 'Kuala Lumpur',
        'W.P. Labuan': 'Labuan',
        'W.P. Putrajaya': 'Putrajaya'
    })
    df_pop_yearly['Population'] = df_pop_yearly['population'] * 1000  # thousands -> actual count

    # External join: accident data (left) with population data (right) on Tahun + Negeri
    state_year = state_year.merge(
        df_pop_yearly[['Tahun', 'Negeri', 'Population']],
        on=['Tahun', 'Negeri'],
        how='left'
    )

    # --- Step 2: Feature Engineering - population-normalized rates ---
    state_year['Accidents_per_100k'] = (state_year['Total_Accidents'] / state_year['Population']) * 100000
    state_year['Fatalities_per_100k'] = (state_year['Fatalities'] / state_year['Population']) * 100000
    state_year = state_year.dropna(subset=['Accidents_per_100k', 'Fatalities_per_100k'])

    state_year.to_csv('outputs/feature_rq3_state_population_normalized.csv', index=False)
    print("[OUTPUT SAVED] outputs/feature_rq3_state_population_normalized.csv")

    # --- Step 3 & 4: Forecast target - Linear Regression per state, per rate ---
    # Fitted ONLY on the pre-pandemic baseline window (train_years).
    future_years = np.array(list(forecast_years)).reshape(-1, 1)
    target_cols = ['Accidents_per_100k', 'Fatalities_per_100k']
    states = sorted(state_year['Negeri'].unique())
    baseline_start, baseline_end = train_years

    metrics_list = []
    future_preds = {'Year': future_years.flatten()}
    full_pivots = {}      # full 2016-2021 actuals, for plotting context
    models_by_key = {}    # fitted model per (state, target), for anchoring the forecast line

    for target in target_cols:
        full_pivot = state_year.pivot(index='Tahun', columns='Negeri', values=target)
        full_pivots[target] = full_pivot

        train_pivot = full_pivot.loc[baseline_start:baseline_end]
        X_train = train_pivot.index.values.reshape(-1, 1)

        for state in states:
            y_train = train_pivot[state].values

            model = LinearRegression()
            model.fit(X_train, y_train)
            models_by_key[(state, target)] = model

            # In-sample evaluation on the baseline window only (not pandemic-distorted)
            y_pred = model.predict(X_train)

            # --- Step 5: Success metrics ---
            metrics_list.append({
                'Negeri': state,
                'Target': target,
                'Train_Window': f'{baseline_start}-{baseline_end}',
                'R2_Score': round(r2_score(y_train, y_pred), 4),
                'RMSE': round(np.sqrt(mean_squared_error(y_train, y_pred)), 2),
                'MAE': round(mean_absolute_error(y_train, y_pred), 2)
            })

            preds = model.predict(future_years)
            future_preds[f'{state}_{target}'] = np.maximum(0, preds.round(2))

    df_metrics = pd.DataFrame(metrics_list)
    df_predictions = pd.DataFrame(future_preds)

    df_metrics.to_csv('outputs/rq3_model_evaluation_metrics.csv', index=False)
    df_predictions.to_csv('outputs/rq3_future_state_rate_predictions.csv', index=False)

    print(f"\n--- Baseline Fit Quality ({baseline_start}-{baseline_end}, pandemic-free) ---")
    print(df_metrics.to_string(index=False))

    print(f"\n--- Forecasted State Rates per 100k Population ({min(forecast_years)}-{max(forecast_years)}) ---")
    for target in target_cols:
        cols = [c for c in df_predictions.columns if c.endswith(target)]
        table = df_predictions[['Year'] + cols].set_index('Year')
        table.columns = [c.replace(f'_{target}', '') for c in table.columns]
        print(f"\n{target}:")
        print(table.T.to_string())

    # --- Visualization: Top 5 states by 2021 rate, Actual (full history) vs Forecast ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    palette = plt.cm.tab10.colors

    for ax, target in zip(axes, target_cols):
        full_pivot = full_pivots[target]
        top_5_states = full_pivot.loc[2021].sort_values(ascending=False).head(5).index
        state_colors = {state: palette[i] for i, state in enumerate(top_5_states)}

        # Shade the pandemic-affected window that was EXCLUDED from training
        ax.axvspan(2020, 2021, color='gray', alpha=0.12, zorder=0)

        # Vertical divider marking the boundary between observed and forecast periods
        ax.axvline(x=(2021 + min(forecast_years)) / 2, color='gray', linestyle=':', linewidth=1.2, zorder=0)

        for state in top_5_states:
            color = state_colors[state]

            # Actual: full observed history 2016-2021 (includes the real pandemic dip, for
            # transparency) plotted as one continuous solid line.
            ax.plot(full_pivot.index, full_pivot[state], color=color, marker='o',
                    linewidth=2, alpha=0.9, zorder=3)

            # Forecast: a SEPARATE line segment anchored at the model's fitted value for
            # the last baseline year (not the actual, pandemic-depressed value), so it
            # continues the pre-pandemic slope cleanly instead of zig-zagging off the dip.
            model = models_by_key[(state, target)]
            anchor_year = baseline_end
            anchor_val = model.predict([[anchor_year]])[0]
            forecast_x = np.concatenate([[anchor_year], future_years.flatten()])
            forecast_y = np.concatenate([[anchor_val], df_predictions[f'{state}_{target}'].values])
            ax.plot(forecast_x, forecast_y, color=color, marker='s', linestyle='--',
                    linewidth=2, alpha=0.55, zorder=2)

        ax.set_title(f'Top 5 States: {target}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(target.replace('_', ' '))

        # Custom legend: state color swatches + a separate Actual/Forecast style key,
        # so 5 states don't produce 10 near-duplicate entries.
        state_handles = [Line2D([0], [0], color=state_colors[s], marker='o', linewidth=2, label=s)
                          for s in top_5_states]
        style_handles = [
            Line2D([0], [0], color='black', linestyle='-', marker='o', label='Actual (2016-2021)'),
            Line2D([0], [0], color='black', linestyle='--', marker='s', alpha=0.55,
                   label=f'Forecast (from {baseline_end} baseline)')
        ]
        ax.legend(handles=state_handles + style_handles, fontsize=7, loc='upper right')

    fig.text(0.5, 0.94,
              f'Pandemic period ({2020}-{2021}) shaded gray: excluded from model training, shown for context only',
              ha='center', fontsize=8.5, style='italic', color='dimgray')
    plt.suptitle('Figure 8: RQ3 Linear Regression Forecast — Population-Normalized State Rates',
                  fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('visualizations/fig8_rq3_ml_predictions.png', dpi=150)
    plt.close()
    print("[PLOT SAVED] visualizations/fig8_rq3_ml_predictions.png")

    return df_metrics, df_predictions

def main():
    #Main execution function.
    setup_directories()
    df_veh, df_inj, df_stat = load_and_clean_data()
    #RQ1
    process_rq1(df_veh)
    ml_predict_rq1(df_veh)
    #RQ2
    process_rq2(df_inj, df_stat)
    ml_predict_rq2(df_stat)
    #RQ3
    process_rq3(df_stat)
    ml_predict_rq3(df_stat)
    print("\n[SUCCESS] Analytical pipeline execution complete. All outputs and figures saved.")

if __name__ == "__main__":
    main()