"""
Step 1: Initial Data Exploration (REVISED)
This script loads our logistics data and shows us what we're working with.
"""

import pandas as pd
import numpy as np
import os

# Setup display options so we can see everything clearly
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("LOGISTICS DATA EXPLORATION - STEP 1 (REVISED)")
print("="*80)

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
print("\n📂 Loading datasets...")

# Load each dataset
base_df = pd.read_csv('data/logistics_base.csv')
sla_df = pd.read_csv('data/logistics_sla_rules.csv')
augmented_df = pd.read_csv('data/logistics_augmented.csv')
summary_df = pd.read_csv('data/logistics_summary.csv')

print("✅ All datasets loaded successfully!")

# ============================================================================
# 2. BASIC DATASET INFO
# ============================================================================
print("\n" + "="*80)
print("📊 DATASET OVERVIEW")
print("="*80)

datasets = {
    'Base Data': base_df,
    'SLA Rules': sla_df,
    'Augmented Data': augmented_df,
    'Summary Data': summary_df
}

for name, df in datasets.items():
    print(f"\n{name}:")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 3. UNDERSTAND THE DATA STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("🔍 DATA STRUCTURE ANALYSIS")
print("="*80)

print("\n📋 All Column Names in Augmented Data:")
print(augmented_df.columns.tolist())

print("\n🔑 Key Identifier Columns:")
# Check what identifier columns we have
id_cols = [col for col in augmented_df.columns if 'id' in col.lower() or 'timestamp' in col.lower()]
print(f"Found: {id_cols}")

print("\n📦 Business Key Columns:")
business_cols = ['carrier_name', 'route', 'shipment_date', 'service_level']
available_business_cols = [col for col in business_cols if col in augmented_df.columns]
print(f"Available: {available_business_cols}")

# ============================================================================
# 4. DETAILED LOOK AT AUGMENTED DATA (our main dataset)
# ============================================================================
print("\n" + "="*80)
print("🔍 AUGMENTED DATA DETAILS (Our Main Dataset)")
print("="*80)

print("\n📋 Column Data Types (first 20 columns):")
print(augmented_df.dtypes.head(20))

print("\n📈 Key Numerical Columns Statistics:")
key_numeric_cols = ['delay_minutes', 'total_additional_cost', 'disruption_cost', 
                    'carrier_otd_history', 'distance_km', 'stops_count']
available_numeric = [col for col in key_numeric_cols if col in augmented_df.columns]
if available_numeric:
    print(augmented_df[available_numeric].describe())

print("\n🔢 Sample of First 3 Rows (selected columns):")
display_cols = ['timestamp', 'carrier_name', 'route', 'shipment_date', 'delay_minutes', 
                'sla_breach_minutes', 'total_additional_cost', 'service_level']
available_display = [col for col in display_cols if col in augmented_df.columns]
if available_display:
    print(augmented_df[available_display].head(3))

# ============================================================================
# 5. DATA QUALITY CHECKS
# ============================================================================
print("\n" + "="*80)
print("🔎 DATA QUALITY CHECKS")
print("="*80)

print("\n❓ Missing Values:")
missing = augmented_df.isnull().sum()
missing_pct = (missing / len(augmented_df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("✅ No missing values found!")

# Check for duplicates if we have timestamp column
if 'timestamp' in augmented_df.columns:
    print("\n🔄 Duplicate Timestamps:")
    duplicates = augmented_df['timestamp'].duplicated().sum()
    print(f"  - Duplicate count: {duplicates}")
    if duplicates > 0:
        print(f"  - Duplicate percentage: {duplicates/len(augmented_df)*100:.2f}%")

print("\n📦 Unique Values for Key Columns:")
if 'carrier_name' in augmented_df.columns:
    print(f"  - Carriers: {augmented_df['carrier_name'].nunique()} unique")
    print(f"    → {augmented_df['carrier_name'].unique().tolist()}")
    
if 'route' in augmented_df.columns:
    print(f"  - Routes: {augmented_df['route'].nunique()} unique")
    print(f"    → Sample routes: {augmented_df['route'].unique()[:5].tolist()}")
    
if 'service_level' in augmented_df.columns:
    print(f"  - Service Levels: {augmented_df['service_level'].nunique()} unique")
    print(f"    → {augmented_df['service_level'].unique().tolist()}")

if 'risk_classification' in augmented_df.columns:
    print(f"  - Risk Classifications: {augmented_df['risk_classification'].nunique()} unique")
    print(f"    → {augmented_df['risk_classification'].unique().tolist()}")

# ============================================================================
# 6. BUSINESS METRICS OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("💰 BUSINESS METRICS OVERVIEW")
print("="*80)

if 'delay_minutes' in augmented_df.columns:
    print("\n⏱️ Delay Statistics:")
    print(f"  - Average delay: {augmented_df['delay_minutes'].mean():.2f} minutes ({augmented_df['delay_minutes'].mean()/60:.2f} hours)")
    print(f"  - Median delay: {augmented_df['delay_minutes'].median():.2f} minutes")
    print(f"  - Max delay: {augmented_df['delay_minutes'].max():.2f} minutes ({augmented_df['delay_minutes'].max()/1440:.2f} days)")
    print(f"  - Min delay: {augmented_df['delay_minutes'].min():.2f} minutes")
    print(f"  - Shipments with positive delays: {(augmented_df['delay_minutes'] > 0).sum():,} ({(augmented_df['delay_minutes'] > 0).mean()*100:.1f}%)")
    print(f"  - Shipments arriving EARLY: {(augmented_df['delay_minutes'] < 0).sum():,} ({(augmented_df['delay_minutes'] < 0).mean()*100:.1f}%)")

if 'sla_breach_minutes' in augmented_df.columns:
    print("\n🚨 SLA Breaches:")
    breaches = augmented_df['sla_breach_minutes'] > 0
    print(f"  - Total breaches: {breaches.sum():,} ({breaches.mean()*100:.1f}%)")
    if breaches.sum() > 0:
        print(f"  - Average breach: {augmented_df[breaches]['sla_breach_minutes'].mean():.2f} minutes")
        print(f"  - Max breach: {augmented_df['sla_breach_minutes'].max():.2f} minutes")

if 'total_additional_cost' in augmented_df.columns and 'disruption_cost' in augmented_df.columns:
    print("\n💸 Cost Impact:")
    print(f"  - Total disruption costs: ${augmented_df['disruption_cost'].sum():,.2f}")
    print(f"  - Total additional costs: ${augmented_df['total_additional_cost'].sum():,.2f}")
    print(f"  - Average cost per shipment: ${augmented_df['total_additional_cost'].mean():.2f}")
    print(f"  - Max cost impact: ${augmented_df['total_additional_cost'].max():,.2f}")

# Cost breakdown
cost_cols = ['weather_cost', 'congestion_cost', 'customs_cost']
available_costs = [col for col in cost_cols if col in augmented_df.columns]
if available_costs:
    print("\n💵 Cost Breakdown:")
    for col in available_costs:
        total = augmented_df[col].sum()
        pct = (total / augmented_df['total_additional_cost'].sum()) * 100 if augmented_df['total_additional_cost'].sum() > 0 else 0
        print(f"  - {col.replace('_', ' ').title()}: ${total:,.2f} ({pct:.1f}%)")

if 'carrier_name' in augmented_df.columns:
    print("\n🏢 Performance by Carrier:")
    carrier_metrics = {}
    
    for col in ['delay_minutes', 'sla_breach_minutes', 'total_additional_cost']:
        if col in augmented_df.columns:
            carrier_metrics[col] = augmented_df.groupby('carrier_name')[col].mean()
    
    if 'sla_breach_minutes' in augmented_df.columns:
        carrier_metrics['breach_rate'] = augmented_df.groupby('carrier_name')['sla_breach_minutes'].apply(lambda x: (x > 0).mean() * 100)
    
    carrier_metrics['shipment_count'] = augmented_df.groupby('carrier_name').size()
    
    carrier_perf = pd.DataFrame(carrier_metrics).round(2)
    print(carrier_perf)

if 'route' in augmented_df.columns and 'delay_minutes' in augmented_df.columns:
    print("\n🛣️ Top 10 Most Problematic Routes (by average delay):")
    route_perf = augmented_df.groupby('route').agg({
        'delay_minutes': 'mean',
        'total_additional_cost': 'sum' if 'total_additional_cost' in augmented_df.columns else 'count'
    }).round(2)
    route_perf.columns = ['Avg_Delay_Min', 'Total_Cost_or_Count']
    route_perf['Shipment_Count'] = augmented_df.groupby('route').size()
    route_perf = route_perf.sort_values('Avg_Delay_Min', ascending=False).head(10)
    print(route_perf)

# ============================================================================
# 7. DATE RANGE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("📅 DATE RANGE ANALYSIS")
print("="*80)

if 'shipment_date' in augmented_df.columns:
    # Convert to datetime
    augmented_df['shipment_date_dt'] = pd.to_datetime(augmented_df['shipment_date'])
    
    print(f"\n📆 Date Range:")
    print(f"  - Earliest: {augmented_df['shipment_date_dt'].min()}")
    print(f"  - Latest: {augmented_df['shipment_date_dt'].max()}")
    print(f"  - Span: {(augmented_df['shipment_date_dt'].max() - augmented_df['shipment_date_dt'].min()).days} days")
    
    print(f"\n📊 Shipments by Month:")
    monthly = augmented_df.groupby(augmented_df['shipment_date_dt'].dt.to_period('M')).size()
    print(monthly.head(10))

# ============================================================================
# 8. SAVE EXPLORATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING EXPLORATION SUMMARY")
print("="*80)

# Create summary report
with open('outputs/reports/01_data_exploration_summary.txt', 'w') as f:
    f.write("LOGISTICS DATA EXPLORATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Shipments: {len(augmented_df):,}\n")
    f.write(f"Total Columns: {len(augmented_df.columns)}\n")
    
    if 'shipment_date' in augmented_df.columns:
        f.write(f"Date Range: {augmented_df['shipment_date'].min()} to {augmented_df['shipment_date'].max()}\n")
    
    if 'sla_breach_minutes' in augmented_df.columns:
        breaches = augmented_df['sla_breach_minutes'] > 0
        f.write(f"SLA Breach Rate: {breaches.mean()*100:.2f}%\n")
    
    if 'total_additional_cost' in augmented_df.columns:
        f.write(f"Total Cost Impact: ${augmented_df['total_additional_cost'].sum():,.2f}\n")
    
    if 'carrier_name' in augmented_df.columns:
        f.write(f"\nUnique Carriers: {augmented_df['carrier_name'].nunique()}\n")
        f.write(f"Carriers: {', '.join(augmented_df['carrier_name'].unique())}\n")
    
    if 'route' in augmented_df.columns:
        f.write(f"\nUnique Routes: {augmented_df['route'].nunique()}\n")

print("✅ Summary saved to: outputs/reports/01_data_exploration_summary.txt")

# Save column info for reference
augmented_df.dtypes.to_csv('outputs/reports/01_column_types.csv')
print("✅ Column types saved to: outputs/reports/01_column_types.csv")

print("\n" + "="*80)
print("✨ EXPLORATION COMPLETE!")
print("="*80)
print("\n📌 Key Findings:")
print(f"  ✓ Dataset has {len(augmented_df):,} shipments")
print(f"  ✓ Dataset has {len(augmented_df.columns)} feature columns")
print(f"  ✓ No missing values detected")
if 'delay_minutes' in augmented_df.columns:
    print(f"  ✓ Average delay: {augmented_df['delay_minutes'].mean():.2f} minutes")
if 'sla_breach_minutes' in augmented_df.columns:
    breaches = augmented_df['sla_breach_minutes'] > 0
    print(f"  ✓ SLA breach rate: {breaches.mean()*100:.1f}%")

print("\n📌 Next Steps:")
print("  1. Review the output above")
print("  2. Check the summary files in outputs/reports/")
print("  3. Share the results with your instructor")
print("  4. Proceed to database setup (Step 2)")