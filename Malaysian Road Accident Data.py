"""
BTPR3203 - Python for Data Science (Project Submission)
Script: main_analysis.py
Title: Analyzing Road Safety Dynamics in Malaysia: Vehicle Involvement, Severity Classifications, and Population-Normalized Geographic Hotspots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set overall plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})

def setup_directories():
    """Ensure output folders exist for processed files and visualizations."""
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    print("[INFO] Output directories initialized.")

def load_and_clean_data():
    """Load raw datasets, clean formatting, and convert data types."""
    print("[INFO] Loading datasets...")
    
    # 1. Load Vehicle Dataset
    df_veh = pd.read_csv('dataset/m-20210325040425_202103250404250_number-of-road-accidents-reported-by-type-of-vehicle-malaysia-2.csv')
    df_veh['Type of Vehicle'] = df_veh['Type of Vehicle'].str.strip()
    
    # 2. Load Injury/Casualty Dataset
    df_inj = pd.read_csv('dataset/m-20210329034857_202307060208460_2000-2021-number-of-deaths-and-injuries-in-road-accidents-repor.csv')
    df_inj['Type of road user'] = df_inj['Type of road user'].str.strip()
    df_inj['Condition'] = df_inj['Condition'].str.strip()
    
    # 3. Load State Statistics Dataset
    df_stat = pd.read_csv('dataset/STATISTIK KEMALANGAN, KEMALANGAN MAUT DAN KEMATIAN.csv')
    df_stat['Negeri'] = df_stat['Negeri'].str.strip()
    df_stat['Klasifikasi Kemalangan'] = df_stat['Klasifikasi Kemalangan'].str.strip()
    
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
    """
    RQ1: Vehicle involvement analysis and longitudinal distribution.
    """
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
    """
    RQ2: Accident classification trends (Deaths vs. Injuries & State Classifications).
    """
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
    
    # State classification national aggregation (2017-2021)
    stat_trend = df_stat.groupby(['Tahun', 'Klasifikasi Kemalangan'])['Jumlah'].sum().unstack()
    stat_trend.to_csv('outputs/processed_national_classification_2017_2021.csv')
    print("[OUTPUT SAVED] outputs/processed_national_classification_2017_2021.csv")
    
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
    plt.title('Figure 3: Total Reported Traffic Investigation Cases in Malaysia (2017–2021)', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Total Cases Opened')
    plt.xticks(stat_trend.index)
    plt.tight_layout()
    plt.savefig('visualizations/fig3_traffic_cases_trend.png')
    plt.close()
    print("[PLOT SAVED] visualizations/fig3_traffic_cases_trend.png")

def process_rq3(df_stat):
    """
    RQ3: State concentration analysis normalized by population metrics using state_population_2019.csv.
    """
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

def main():
    """Main execution function."""
    setup_directories()
    df_veh, df_inj, df_stat = load_and_clean_data()
    process_rq1(df_veh)
    process_rq2(df_inj, df_stat)
    process_rq3(df_stat)
    print("\n[SUCCESS] Analytical pipeline execution complete. All outputs and figures saved.")

if __name__ == "__main__":
    main()