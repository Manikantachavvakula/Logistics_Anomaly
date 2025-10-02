"""
Step 3: Feature Engineering for Anomaly Detection
This script creates new features that will help our model detect anomalies.
"""


import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FEATURE ENGINEERING - STEP 3")
print("="*80)

# ============================================================================
# 1. LOAD DATA FROM DATABASE
# ============================================================================
print("\n📂 Loading data from database...")

conn = sqlite3.connect('data/logistics_analytics.db')

# Load main shipments table
df = pd.read_sql_query("SELECT * FROM shipments", conn)
print(f"  ✅ Loaded {len(df):,} shipments")

# Convert date columns back to datetime
date_cols = ['shipment_date', 'planned_delivery', 'actual_delivery']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

print(f"  ✅ Converted date columns to datetime")

# ============================================================================
# 2. BASIC DERIVED FEATURES
# ============================================================================
print("\n🔧 Creating basic derived features...")

# 2.1 Time-based features
df['shipment_year'] = df['shipment_date'].dt.year
df['shipment_month'] = df['shipment_date'].dt.month
df['shipment_day'] = df['shipment_date'].dt.day
df['shipment_dayofweek'] = df['shipment_date'].dt.dayofweek
df['shipment_weekofyear'] = df['shipment_date'].dt.isocalendar().week

print(f"  ✅ Created time-based features (5 features)")

# 2.2 Delay categorization
df['delay_category'] = pd.cut(
    df['delay_minutes'],
    bins=[-np.inf, 0, 1440, 2880, 7200, np.inf],
    labels=['Early/On-Time', 'Minor_Delay', 'Major_Delay', 'Critical_Delay', 'Extreme_Delay']
)

# Binary flags
df['is_delayed'] = (df['delay_minutes'] > 0).astype(int)
df['is_sla_breach'] = (df['sla_breach_minutes'] > 0).astype(int)
df['is_early'] = (df['delay_minutes'] < 0).astype(int)

print(f"  ✅ Created delay categorization features (4 features)")

# 2.3 Cost-related features
df['cost_per_km'] = df['total_additional_cost'] / (df['distance_km'] + 1)  # +1 to avoid division by zero
df['cost_per_kg'] = df['total_additional_cost'] / (df['weight_kg'] + 1)
df['cargo_value_density'] = df['cargo_value'] / (df['weight_kg'] + 1)

# Cost flags
df['has_weather_cost'] = (df['weather_cost'] > 0).astype(int)
df['has_congestion_cost'] = (df['congestion_cost'] > 0).astype(int)
df['has_customs_cost'] = (df['customs_cost'] > 0).astype(int)
df['has_disruption_cost'] = (df['disruption_cost'] > 0).astype(int)

print(f"  ✅ Created cost-related features (7 features)")

# ============================================================================
# 3. CARRIER PERFORMANCE FEATURES
# ============================================================================
print("\n📊 Creating carrier performance features...")

# 3.1 Carrier aggregations
carrier_stats = df.groupby('carrier_name').agg({
    'delay_minutes': ['mean', 'std', 'median'],
    'sla_breach_minutes': 'mean',
    'total_additional_cost': 'mean',
    'carrier_otd_history': 'mean'
}).round(2)

# Flatten column names
carrier_stats.columns = ['_'.join(col).strip() for col in carrier_stats.columns.values]
carrier_stats = carrier_stats.add_prefix('carrier_')
carrier_stats = carrier_stats.reset_index()

# Merge back to main dataframe
df = df.merge(carrier_stats, on='carrier_name', how='left')

print(f"  ✅ Created carrier aggregation features (6 features)")

# 3.2 Deviation from carrier average
df['delay_vs_carrier_avg'] = df['delay_minutes'] - df['carrier_delay_minutes_mean']
df['cost_vs_carrier_avg'] = df['total_additional_cost'] - df['carrier_total_additional_cost_mean']

print(f"  ✅ Created carrier deviation features (2 features)")

# ============================================================================
# 4. ROUTE PERFORMANCE FEATURES
# ============================================================================
print("\n🛣️ Creating route performance features...")

# 4.1 Route aggregations
route_stats = df.groupby('route').agg({
    'delay_minutes': ['mean', 'std', 'median'],
    'distance_km': 'mean',
    'total_additional_cost': 'mean',
    'sla_breach_minutes': 'mean'
}).round(2)

route_stats.columns = ['_'.join(col).strip() for col in route_stats.columns.values]
route_stats = route_stats.add_prefix('route_')
route_stats = route_stats.reset_index()

df = df.merge(route_stats, on='route', how='left')

print(f"  ✅ Created route aggregation features (6 features)")

# 4.2 Deviation from route average
df['delay_vs_route_avg'] = df['delay_minutes'] - df['route_delay_minutes_mean']
df['cost_vs_route_avg'] = df['total_additional_cost'] - df['route_total_additional_cost_mean']

print(f"  ✅ Created route deviation features (2 features)")

# ============================================================================
# 5. SERVICE LEVEL FEATURES
# ============================================================================
print("\n⚡ Creating service level features...")

# Service level aggregations
service_stats = df.groupby('service_level').agg({
    'delay_minutes': ['mean', 'std'],
    'total_additional_cost': 'mean',
    'sla_breach_minutes': 'mean'
}).round(2)

service_stats.columns = ['_'.join(col).strip() for col in service_stats.columns.values]
service_stats = service_stats.add_prefix('service_')
service_stats = service_stats.reset_index()

df = df.merge(service_stats, on='service_level', how='left')

print(f"  ✅ Created service level features (4 features)")

# ============================================================================
# 6. TEMPORAL AGGREGATION FEATURES
# ============================================================================
print("\n📅 Creating temporal aggregation features...")

# Monthly aggregations
monthly_stats = df.groupby('month').agg({
    'delay_minutes': ['mean', 'std'],
    'sla_breach_minutes': 'mean'
}).round(2)

monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
monthly_stats = monthly_stats.add_prefix('monthly_')
monthly_stats = monthly_stats.reset_index()

df = df.merge(monthly_stats, on='month', how='left')

print(f"  ✅ Created monthly aggregation features (3 features)")

# ============================================================================
# 7. RATIO AND EFFICIENCY FEATURES
# ============================================================================
print("\n📈 Creating ratio and efficiency features...")

# Delay ratios
df['delay_to_distance_ratio'] = df['delay_minutes'] / (df['distance_km'] + 1)
df['delay_to_sla_ratio'] = df['delay_minutes'] / (df['sla_minutes'] + 1)
df['breach_to_sla_ratio'] = df['sla_breach_minutes'] / (df['sla_minutes'] + 1)

# Weather/congestion impact ratios
df['weather_delay_ratio'] = df['weather_delay_minutes'] / (df['delay_minutes'] + 1)
df['port_delay_ratio'] = df['port_congestion_minutes'] / (df['delay_minutes'] + 1)
df['customs_delay_ratio'] = df['customs_delay_minutes'] / (df['delay_minutes'] + 1)

# Cost efficiency
df['cost_efficiency'] = df['cargo_value'] / (df['total_additional_cost'] + 1)

print(f"  ✅ Created ratio features (7 features)")

# ============================================================================
# 8. ANOMALY SCORE FEATURES (Pre-computed indicators)
# ============================================================================
print("\n🎯 Creating anomaly indicator features...")

# Z-score normalization for key metrics
from scipy import stats

# Calculate z-scores for important columns
z_score_cols = ['delay_minutes', 'total_additional_cost', 'sla_breach_minutes', 
                'distance_km', 'cargo_value']

for col in z_score_cols:
    if col in df.columns:
        df[f'{col}_zscore'] = np.abs(stats.zscore(df[col], nan_policy='omit'))

print(f"  ✅ Created z-score features (5 features)")

# Composite anomaly indicators
df['extreme_delay_flag'] = (df['delay_minutes'] > df['delay_minutes'].quantile(0.95)).astype(int)
df['extreme_cost_flag'] = (df['total_additional_cost'] > df['total_additional_cost'].quantile(0.95)).astype(int)
df['multiple_delay_causes'] = (
    df['has_weather_cost'] + 
    df['has_congestion_cost'] + 
    df['has_customs_cost']
)

print(f"  ✅ Created composite anomaly indicators (3 features)")

# ============================================================================
# 9. INTERACTION FEATURES
# ============================================================================
print("\n🔗 Creating interaction features...")

# Important interactions for anomaly detection
df['carrier_route_risk'] = df['carrier_delay_minutes_mean'] * df['route_complexity']
df['service_distance_interaction'] = df['service_level'].map({
    'Standard': 1, 'Priority': 2, 'Express': 3
}) * df['distance_km']

df['weekend_delay_interaction'] = df['is_weekend'].astype(int) * df['delay_minutes']

print(f"  ✅ Created interaction features (3 features)")

# ============================================================================
# 10. HANDLE MISSING VALUES
# ============================================================================
print("\n🔍 Checking for missing values...")

missing_before = df.isnull().sum().sum()
print(f"  ℹ️  Missing values before: {missing_before}")

# Fill any NaN values created during feature engineering
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

missing_after = df.isnull().sum().sum()
print(f"  ✅ Missing values after: {missing_after}")

# ============================================================================
# 11. FEATURE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FEATURE ENGINEERING SUMMARY")
print("="*80)

original_cols = 54  # From exploration step
new_cols = len(df.columns)
engineered_features = new_cols - original_cols

print(f"\n  📈 Feature Statistics:")
print(f"     - Original features: {original_cols}")
print(f"     - New engineered features: {engineered_features}")
print(f"     - Total features: {new_cols}")

# List new feature categories
print(f"\n  🎯 Feature Categories:")
print(f"     - Time-based: 5 features")
print(f"     - Delay categorization: 4 features")
print(f"     - Cost-related: 7 features")
print(f"     - Carrier performance: 8 features")
print(f"     - Route performance: 8 features")
print(f"     - Service level: 4 features")
print(f"     - Temporal: 3 features")
print(f"     - Ratios: 7 features")
print(f"     - Anomaly indicators: 8 features")
print(f"     - Interactions: 3 features")

# ============================================================================
# 12. SAVE ENGINEERED FEATURES
# ============================================================================
print("\n💾 Saving engineered features...")

# Save to CSV
output_file = 'data/logistics_features_engineered.csv'
df.to_csv(output_file, index=False)
print(f"  ✅ Saved to: {output_file}")
print(f"  ✅ File size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

# Update database with new table
df.to_sql('shipments_engineered', conn, if_exists='replace', index=False)
print(f"  ✅ Updated database with 'shipments_engineered' table")

# ============================================================================
# 13. CREATE FEATURE LIST FOR MODELING
# ============================================================================
print("\n📋 Creating feature list for modeling...")

# Identify features for anomaly detection
# Exclude ID, dates, and categorical text columns
exclude_cols = [
    'shipment_id', 'timestamp', 'shipment_date', 'planned_delivery', 
    'actual_delivery', 'carrier_name', 'route', 'carrier', 
    'service_level', 'risk_classification', 'delay_category'
]

# Get numeric columns for modeling
modeling_features = [col for col in df.columns if col not in exclude_cols 
                     and df[col].dtype in ['int64', 'float64', 'bool']]

print(f"  ✅ Identified {len(modeling_features)} features for modeling")

# Save feature list
feature_list = pd.DataFrame({
    'feature_name': modeling_features,
    'data_type': [str(df[col].dtype) for col in modeling_features]
})

feature_list.to_csv('outputs/reports/03_modeling_features.csv', index=False)
print(f"  ✅ Saved feature list to: outputs/reports/03_modeling_features.csv")

# ============================================================================
# 14. SAMPLE DATA INSPECTION
# ============================================================================
print("\n🔍 Sample of engineered features:")
print("\nNew features (first 10):")
new_feature_cols = [col for col in df.columns if col not in 
                    ['shipment_id', 'timestamp', 'vehicle_gps_latitude', 
                     'vehicle_gps_longitude', 'fuel_consumption_rate']][:10]
print(df[new_feature_cols].head(3))

# ============================================================================
# 15. CLEANUP
# ============================================================================
conn.close()

print("\n" + "="*80)
print("✨ FEATURE ENGINEERING COMPLETE!")
print("="*80)
print("\n📌 Summary:")
print(f"  ✓ Created {engineered_features} new features")
print(f"  ✓ Total features: {new_cols}")
print(f"  ✓ Features ready for modeling: {len(modeling_features)}")
print(f"  ✓ Data saved to: {output_file}")
print("\n📌 Next Steps:")
print("  1. Review the new features in the CSV file")
print("  2. Check outputs/reports/03_modeling_features.csv")
print("  3. Share results with instructor")
print("  4. Proceed to Anomaly Detection (Step 4)")
