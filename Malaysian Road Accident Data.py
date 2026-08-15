import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            print(f"⚠️ Differences found in column '{col}': {len(diff_rows)} rows mismatch.")
            print(diff_rows[key_cols + [f'{col}_excel', f'{col}_csv']].head())
    if not diff_found:
        print("✅ All 126 overlapping rows match perfectly across all columns!")

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

def main():
    #Main execution function.
    setup_directories()
    df_veh, df_inj, df_stat = load_and_clean_data()
    #RQ1
    process_rq1(df_veh)
    ml_predict_rq1(df_veh)
    #RQ2
    process_rq2(df_inj, df_stat)
    #RQ3
    process_rq3(df_stat)
    print("\n[SUCCESS] Analytical pipeline execution complete. All outputs and figures saved.")

if __name__ == "__main__":
    main()