"""
Step 2: Database Setup and Data Loading
This script creates a SQLite database and loads our logistics data into it.
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime

print("="*80)
print("DATABASE SETUP - STEP 2")
print("="*80)

# ============================================================================
# 1. CREATE DATABASE CONNECTION
# ============================================================================
print("\n📁 Setting up database...")

# Database will be stored in data folder
db_path = 'data/logistics_analytics.db'

# Remove old database if exists (fresh start)
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"  ⚠️  Removed existing database")

# Create new connection
conn = sqlite3.connect(db_path)
print(f"  ✅ Created database: {db_path}")

# ============================================================================
# 2. LOAD DATA FROM CSV FILES
# ============================================================================
print("\n📂 Loading CSV files...")

# Load all datasets
base_df = pd.read_csv('data/logistics_base.csv')
sla_df = pd.read_csv('data/logistics_sla_rules.csv')
augmented_df = pd.read_csv('data/logistics_augmented.csv')
summary_df = pd.read_csv('data/logistics_summary.csv')

print(f"  ✅ Loaded base data: {len(base_df):,} rows")
print(f"  ✅ Loaded SLA rules: {len(sla_df):,} rows")
print(f"  ✅ Loaded augmented data: {len(augmented_df):,} rows")
print(f"  ✅ Loaded summary data: {len(summary_df):,} rows")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing data...")

# Convert date columns to proper datetime format
date_columns = ['shipment_date', 'planned_delivery', 'actual_delivery']
for col in date_columns:
    if col in augmented_df.columns:
        augmented_df[col] = pd.to_datetime(augmented_df[col])
        print(f"  ✅ Converted {col} to datetime")

# Add a unique ID for each shipment (using row index as ID)
augmented_df['shipment_id'] = range(1, len(augmented_df) + 1)
print(f"  ✅ Added shipment_id column")

# Reorder columns to put ID first
cols = ['shipment_id'] + [col for col in augmented_df.columns if col != 'shipment_id']
augmented_df = augmented_df[cols]

# ============================================================================
# 4. CREATE DATABASE TABLES
# ============================================================================
print("\n📊 Creating database tables...")

# Table 1: Main shipments table (augmented data)
augmented_df.to_sql('shipments', conn, if_exists='replace', index=False)
print(f"  ✅ Created 'shipments' table: {len(augmented_df):,} rows, {len(augmented_df.columns)} columns")

# Table 2: SLA rules lookup table
sla_df.to_sql('sla_rules', conn, if_exists='replace', index=False)
print(f"  ✅ Created 'sla_rules' table: {len(sla_df):,} rows")

# Table 3: Summary table (for quick queries)
summary_df.to_sql('shipments_summary', conn, if_exists='replace', index=False)
print(f"  ✅ Created 'shipments_summary' table: {len(summary_df):,} rows")

# ============================================================================
# 5. CREATE INDEXES FOR PERFORMANCE
# ============================================================================
print("\n⚡ Creating indexes for fast queries...")

# Create indexes on commonly queried columns
indexes = [
    "CREATE INDEX idx_carrier ON shipments(carrier_name)",
    "CREATE INDEX idx_route ON shipments(route)",
    "CREATE INDEX idx_shipment_date ON shipments(shipment_date)",
    "CREATE INDEX idx_service_level ON shipments(service_level)",
    "CREATE INDEX idx_sla_breach ON shipments(sla_breach_minutes)",
    "CREATE INDEX idx_delay ON shipments(delay_minutes)",
]

cursor = conn.cursor()
for idx_sql in indexes:
    cursor.execute(idx_sql)
    index_name = idx_sql.split()[2]  # Extract index name
    print(f"  ✅ Created {index_name}")

conn.commit()

# ============================================================================
# 6. VERIFY DATABASE TABLES
# ============================================================================
print("\n🔍 Verifying database tables...")

# Get list of all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\n  📋 Tables in database:")
for table in tables:
    print(f"     - {table[0]}")

# Get row counts for each table
print(f"\n  📊 Row counts:")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"     - {table[0]}: {count:,} rows")

# ============================================================================
# 7. TEST QUERIES
# ============================================================================
print("\n" + "="*80)
print("🧪 TESTING DATABASE QUERIES")
print("="*80)

# Test Query 1: Total shipments by carrier
print("\n📊 Test Query 1: Shipments by Carrier")
query1 = """
SELECT 
    carrier_name,
    COUNT(*) as shipment_count,
    ROUND(AVG(delay_minutes), 2) as avg_delay_min,
    SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) as breach_count
FROM shipments
GROUP BY carrier_name
ORDER BY breach_count DESC
"""
result1 = pd.read_sql_query(query1, conn)
print(result1)

# Test Query 2: Most expensive delays
print("\n💰 Test Query 2: Top 5 Most Expensive Shipments")
query2 = """
SELECT 
    shipment_id,
    carrier_name,
    route,
    delay_minutes,
    sla_breach_minutes,
    total_additional_cost,
    shipment_date
FROM shipments
ORDER BY total_additional_cost DESC
LIMIT 5
"""
result2 = pd.read_sql_query(query2, conn)
print(result2)

# Test Query 3: Monthly trend
print("\n📈 Test Query 3: Monthly Shipment Trends")
query3 = """
SELECT 
    strftime('%Y-%m', shipment_date) as month,
    COUNT(*) as shipments,
    ROUND(AVG(delay_minutes), 2) as avg_delay,
    SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) as breaches
FROM shipments
GROUP BY strftime('%Y-%m', shipment_date)
ORDER BY month
LIMIT 10
"""
result3 = pd.read_sql_query(query3, conn)
print(result3)

# ============================================================================
# 8. CREATE SQL SCRIPTS FOR POWER BI
# ============================================================================
print("\n" + "="*80)
print("💾 CREATING SQL SCRIPTS FOR POWER BI")
print("="*80)

# Create SQL folder if doesn't exist
os.makedirs('sql', exist_ok=True)

# Script 1: Main dashboard query
dashboard_query = """
-- Main Dashboard Query
-- Use this in Power BI to load shipment data

SELECT 
    shipment_id,
    timestamp,
    carrier_name,
    route,
    service_level,
    shipment_date,
    planned_delivery,
    actual_delivery,
    delay_minutes,
    sla_minutes,
    sla_breach_minutes,
    disruption_cost,
    weather_cost,
    congestion_cost,
    customs_cost,
    total_additional_cost,
    cargo_value,
    weight_kg,
    distance_km,
    stops_count,
    carrier_otd_history,
    weather_delay_minutes,
    port_congestion_minutes,
    customs_delay_minutes,
    risk_classification,
    month,
    quarter,
    is_weekend,
    route_complexity,
    -- Derived fields
    CASE 
        WHEN sla_breach_minutes > 0 THEN 'Breach'
        ELSE 'On-Time'
    END as sla_status,
    CASE 
        WHEN delay_minutes > 2880 THEN 'Critical'
        WHEN delay_minutes > 1440 THEN 'Major'
        WHEN delay_minutes > 0 THEN 'Minor'
        ELSE 'On-Time'
    END as delay_category
FROM shipments
ORDER BY shipment_date DESC;
"""

with open('sql/dashboard_query.sql', 'w') as f:
    f.write(dashboard_query)
print("  ✅ Created sql/dashboard_query.sql")

# Script 2: Carrier performance query
carrier_query = """
-- Carrier Performance Analysis
-- Use this to analyze carrier metrics

SELECT 
    carrier_name,
    COUNT(*) as total_shipments,
    ROUND(AVG(delay_minutes), 2) as avg_delay_minutes,
    ROUND(AVG(carrier_otd_history), 2) as avg_otd_score,
    SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) as total_breaches,
    ROUND(100.0 * SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as breach_rate_pct,
    ROUND(SUM(total_additional_cost), 2) as total_cost_impact,
    ROUND(AVG(total_additional_cost), 2) as avg_cost_per_shipment
FROM shipments
GROUP BY carrier_name
ORDER BY breach_rate_pct DESC;
"""

with open('sql/carrier_performance.sql', 'w') as f:
    f.write(carrier_query)
print("  ✅ Created sql/carrier_performance.sql")

# Script 3: Route analysis query
route_query = """
-- Route Analysis
-- Analyze performance by route

SELECT 
    route,
    COUNT(*) as shipment_count,
    ROUND(AVG(distance_km), 2) as avg_distance_km,
    ROUND(AVG(delay_minutes), 2) as avg_delay_minutes,
    ROUND(AVG(route_complexity), 3) as avg_complexity,
    SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) as breaches,
    ROUND(SUM(total_additional_cost), 2) as total_cost
FROM shipments
GROUP BY route
ORDER BY total_cost DESC;
"""

with open('sql/route_analysis.sql', 'w') as f:
    f.write(route_query)
print("  ✅ Created sql/route_analysis.sql")

# ============================================================================
# 9. SAVE DATABASE SCHEMA
# ============================================================================
print("\n📋 Saving database schema...")

# Get schema for all tables
schema_info = []
for table in tables:
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    schema_info.append(f"\n{'='*60}\nTable: {table[0]}\n{'='*60}")
    for col in columns:
        schema_info.append(f"  {col[1]:30} {col[2]:15}")

schema_text = "\n".join(schema_info)

with open('outputs/reports/02_database_schema.txt', 'w') as f:
    f.write("DATABASE SCHEMA\n")
    f.write("="*80 + "\n")
    f.write(f"Database: {db_path}\n")
    f.write(f"Created: {datetime.now()}\n")
    f.write(schema_text)

print(f"  ✅ Schema saved to: outputs/reports/02_database_schema.txt")

# ============================================================================
# 10. CLEANUP
# ============================================================================
conn.close()
print("\n✅ Database connection closed")

print("\n" + "="*80)
print("✨ DATABASE SETUP COMPLETE!")
print("="*80)
print(f"\n📌 Database Location: {db_path}")
print(f"📌 Database Size: {os.path.getsize(db_path) / 1024**2:.2f} MB")
print("\n📌 Next Steps:")
print("  1. Verify database was created in data/ folder")
print("  2. Check the SQL scripts in sql/ folder")
print("  3. Review database schema in outputs/reports/")
print("  4. Share results with instructor")
print("  5. Proceed to Feature Engineering (Step 3)")