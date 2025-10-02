"""
STEP 6: Power BI Dashboard Data Preparation
This script prepares optimized datasets for Power BI dashboard creation.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime

print("="*80)
print("POWER BI DASHBOARD PREPARATION - STEP 6")
print("="*80)

def prepare_powerbi_data():
    # ============================================================================
    # 1. LOAD ALL REQUIRED DATA
    # ============================================================================
    print("\n📂 Loading data for Power BI...")
    
    # Load main data with anomalies
    df = pd.read_csv('data/logistics_with_anomalies.csv')
    print(f"  ✅ Loaded {len(df):,} shipments with anomaly predictions")
    
    # Load additional data if available
    try:
        conn = sqlite3.connect('data/logistics_analytics.db')
        # Try to load enhanced data
        enhanced_df = pd.read_sql_query("SELECT * FROM shipments_engineered LIMIT 1", conn)
        print(f"  ✅ Database connection successful")
        conn.close()
    except:
        print(f"  ⚠️  Using basic dataset only")
    
    # ============================================================================
    # 2. CREATE POWER BI OPTIMIZED DATASETS
    # ============================================================================
    print("\n📊 Creating Power BI optimized datasets...")
    
    # Dataset 1: Main Dashboard Data
    print("\n🔹 Creating Main Dashboard Dataset...")
    
    # Add calculated fields for Power BI
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    df['month_name'] = df['shipment_date'].dt.month_name()
    df['year'] = df['shipment_date'].dt.year
    df['quarter'] = df['shipment_date'].dt.quarter
    df['day_of_week'] = df['shipment_date'].dt.day_name()
    
    # Create categories for better visualization
    df['delay_category'] = pd.cut(df['delay_minutes'], 
                                 bins=[-1, 0, 60, 1440, 2880, float('inf')],
                                 labels=['On-Time', 'Minor (<1h)', 'Moderate (<1d)', 'Major (<2d)', 'Critical (2d+)'])
    
    df['cost_category'] = pd.cut(df['total_additional_cost'],
                                bins=[-1, 0, 100, 1000, 5000, float('inf')],
                                labels=['No Cost', 'Low ($0-100)', 'Medium ($100-1k)', 'High ($1k-5k)', 'Critical ($5k+)'])
    
    df['sla_status'] = np.where(df['sla_breach_minutes'] > 0, 'SLA Breach', 'On-Time')
    
    # Save main dataset
    powerbi_cols = ['shipment_id', 'carrier_name', 'route', 'shipment_date', 'month_name', 
                   'year', 'quarter', 'day_of_week', 'delay_minutes', 'delay_category',
                   'sla_breach_minutes', 'sla_status', 'total_additional_cost', 'cost_category',
                   'is_anomaly', 'anomaly_score', 'risk_category']
    
    powerbi_df = df[powerbi_cols].copy()
    powerbi_df.to_csv('powerbi/logistics_dashboard_data.csv', index=False)
    print(f"  ✅ Main dashboard data: {len(powerbi_df):,} rows, {len(powerbi_df.columns)} columns")
    
    # ============================================================================
    # 3. CREATE AGGREGATED DATASETS FOR PERFORMANCE
    # ============================================================================
    print("\n🔹 Creating Aggregated Summary Datasets...")
    
    # Dataset 2: Carrier Performance Summary
    carrier_summary = df.groupby('carrier_name').agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': ['mean', 'max'],
        'sla_breach_minutes': ['mean', 'sum'],
        'total_additional_cost': ['sum', 'mean']
    }).round(2)
    
    carrier_summary.columns = ['total_shipments', 'anomaly_count', 'avg_delay', 'max_delay',
                              'avg_breach', 'total_breach_minutes', 'total_cost', 'avg_cost']
    
    carrier_summary['anomaly_rate'] = (carrier_summary['anomaly_count'] / carrier_summary['total_shipments'] * 100).round(1)
    carrier_summary['on_time_rate'] = (100 - carrier_summary['anomaly_rate']).round(1)
    carrier_summary['cost_per_shipment'] = (carrier_summary['total_cost'] / carrier_summary['total_shipments']).round(2)
    
    carrier_summary.reset_index(inplace=True)
    carrier_summary.to_csv('powerbi/carrier_performance_summary.csv', index=False)
    print(f"  ✅ Carrier performance data: {len(carrier_summary):,} carriers")
    
    # Dataset 3: Route Performance Summary
    route_summary = df.groupby('route').agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': 'mean',
        'total_additional_cost': 'sum',
        'sla_breach_minutes': 'mean'
    }).round(2)
    
    route_summary.columns = ['total_shipments', 'anomaly_count', 'avg_delay', 'total_cost', 'avg_breach']
    route_summary['anomaly_rate'] = (route_summary['anomaly_count'] / route_summary['total_shipments'] * 100).round(1)
    
    route_summary.reset_index(inplace=True)
    route_summary.to_csv('powerbi/route_performance_summary.csv', index=False)
    print(f"  ✅ Route performance data: {len(route_summary):,} routes")
    
    # Dataset 4: Monthly Trends
    monthly_trends = df.groupby(['year', 'month_name']).agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': 'mean',
        'total_additional_cost': 'sum'
    }).round(2)
    
    monthly_trends.columns = ['total_shipments', 'anomaly_count', 'avg_delay', 'total_cost']
    monthly_trends['anomaly_rate'] = (monthly_trends['anomaly_count'] / monthly_trends['total_shipments'] * 100).round(1)
    
    monthly_trends.reset_index(inplace=True)
    monthly_trends.to_csv('powerbi/monthly_trends.csv', index=False)
    print(f"  ✅ Monthly trends data: {len(monthly_trends):,} months")
    
    # Dataset 5: Risk Analysis
    risk_analysis = df.groupby('risk_category').agg({
        'shipment_id': 'count',
        'delay_minutes': 'mean',
        'total_additional_cost': ['sum', 'mean'],
        'sla_breach_minutes': 'mean'
    }).round(2)
    
    risk_analysis.columns = ['shipment_count', 'avg_delay', 'total_cost', 'avg_cost', 'avg_breach']
    risk_analysis['cost_percentage'] = (risk_analysis['total_cost'] / risk_analysis['total_cost'].sum() * 100).round(1)
    
    risk_analysis.reset_index(inplace=True)
    risk_analysis.to_csv('powerbi/risk_analysis.csv', index=False)
    print(f"  ✅ Risk analysis data: {len(risk_analysis):,} risk categories")
    
    # ============================================================================
    # 4. CREATE TOP 10 LISTS FOR DASHBOARD
    # ============================================================================
    print("\n🔹 Creating Top 10 Lists...")
    
    # Top 10 Most Expensive Anomalies
    top_anomalies = df[df['is_anomaly'] == 1].nlargest(10, 'total_additional_cost')[
        ['shipment_id', 'carrier_name', 'route', 'shipment_date', 'delay_minutes', 'total_additional_cost', 'risk_category']
    ]
    top_anomalies.to_csv('powerbi/top_10_expensive_anomalies.csv', index=False)
    print(f"  ✅ Top 10 expensive anomalies: ${top_anomalies['total_additional_cost'].sum():,.2f} total")
    
    # Top 10 Longest Delays
    top_delays = df.nlargest(10, 'delay_minutes')[
        ['shipment_id', 'carrier_name', 'route', 'shipment_date', 'delay_minutes', 'total_additional_cost', 'is_anomaly']
    ]
    top_delays.to_csv('powerbi/top_10_longest_delays.csv', index=False)
    print(f"  ✅ Top 10 longest delays: {top_delays['delay_minutes'].max():.0f} minutes max")
    
    # ============================================================================
    # 5. CREATE DASHBOARD METRICS SUMMARY
    # ============================================================================
    print("\n🔹 Creating Dashboard Metrics Summary...")
    
    metrics_summary = {
        'total_shipments': len(df),
        'total_anomalies': df['is_anomaly'].sum(),
        'anomaly_rate': (df['is_anomaly'].sum() / len(df) * 100),
        'total_additional_costs': df['total_additional_cost'].sum(),
        'avg_delay_minutes': df['delay_minutes'].mean(),
        'sla_breach_count': (df['sla_breach_minutes'] > 0).sum(),
        'sla_breach_rate': ((df['sla_breach_minutes'] > 0).sum() / len(df) * 100),
        'critical_risk_count': (df['risk_category'] == 'Critical').sum() if 'risk_category' in df.columns else 0,
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv('powerbi/dashboard_metrics.csv', index=False)
    print(f"  ✅ Dashboard metrics created")
    
    # ============================================================================
    # 6. CREATE POWER BI SETUP INSTRUCTIONS
    # ============================================================================
    print("\n🔹 Creating Power BI Setup Guide...")
    
    setup_guide = f"""
POWER BI DASHBOARD SETUP GUIDE
==============================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASETS CREATED:
-----------------
1. logistics_dashboard_data.csv ({len(powerbi_df):,} rows)
   - Main dataset for detailed analysis
   - Use for drill-through and detailed views

2. carrier_performance_summary.csv ({len(carrier_summary):,} rows)
   - Pre-aggregated carrier metrics
   - Use for carrier comparison charts

3. route_performance_summary.csv ({len(route_summary):,} rows)
   - Pre-aggregated route metrics  
   - Use for route analysis maps/charts

4. monthly_trends.csv ({len(monthly_trends):,} rows)
   - Monthly aggregated trends
   - Use for time series analysis

5. risk_analysis.csv ({len(risk_analysis):,} rows)
   - Risk category breakdown
   - Use for risk distribution charts

6. top_10_expensive_anomalies.csv (10 rows)
   - Most costly problematic shipments
   - Use for management attention

7. top_10_longest_delays.csv (10 rows)
   - Longest delay incidents
   - Use for operational review

8. dashboard_metrics.csv (1 row)
   - Key performance indicators
   - Use for summary cards

RECOMMENDED DASHBOARD STRUCTURE:
--------------------------------
1. EXECUTIVE SUMMARY PAGE
   - KPI Cards: Total shipments, anomaly rate, total costs
   - Carrier performance comparison
   - Monthly trend chart

2. CARRIER ANALYTICS PAGE  
   - Carrier ranking by anomaly rate
   - Cost analysis by carrier
   - Delay patterns by carrier

3. ROUTE ANALYTICS PAGE
   - Route performance map
   - Route comparison charts
   - Geographic hot spots

4. ANOMALY DETAILS PAGE
   - Anomaly breakdown by risk category
   - Top expensive anomalies list
   - Root cause analysis

5. TEMPORAL ANALYSIS PAGE
   - Monthly/quarterly trends
   - Weekend vs weekday performance
   - Seasonal patterns

IMPORTANT NOTES:
----------------
- All data is pre-cleaned and optimized for Power BI
- Relationships: Use carrier_name and route as keys
- Refresh schedule: Daily recommended
- File location: ./powerbi/ folder

NEXT STEPS:
-----------
1. Open Power BI Desktop
2. Import these CSV files
3. Create relationships between tables
4. Build visualizations following the structure above
5. Publish to Power BI Service for sharing
"""

    with open('powerbi/POWER_BI_SETUP_GUIDE.txt', 'w') as f:
        f.write(setup_guide)
    
    print("  ✅ Power BI setup guide created")
    
    # ============================================================================
    # 7. FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("✨ POWER BI DATA PREPARATION COMPLETE!")
    print("="*80)
    
    total_files = len([f for f in os.listdir('powerbi') if f.endswith('.csv')])
    total_size = sum(os.path.getsize(os.path.join('powerbi', f)) for f in os.listdir('powerbi') if f.endswith('.csv')) / (1024*1024)
    
    print(f"\n📊 FILES CREATED IN powerbi/ FOLDER:")
    print(f"   • {total_files} dataset files")
    print(f"   • {total_size:.2f} MB total data")
    print(f"   • {len(powerbi_df):,} detailed shipment records")
    print(f"   • {len(carrier_summary):,} carrier summaries") 
    print(f"   • {len(route_summary):,} route summaries")
    print(f"   • Setup guide: powerbi/POWER_BI_SETUP_GUIDE.txt")
    
    print(f"\n🎯 KEY METRICS FOR DASHBOARD:")
    print(f"   • Total Shipments: {metrics_summary['total_shipments']:,}")
    print(f"   • Anomaly Rate: {metrics_summary['anomaly_rate']:.1f}%")
    print(f"   • Total Costs: ${metrics_summary['total_additional_costs']:,.2f}")
    print(f"   • SLA Breach Rate: {metrics_summary['sla_breach_rate']:.1f}%")
    
    print(f"\n📌 NEXT STEPS:")
    print(f"   1. Review files in powerbi/ folder")
    print(f"   2. Follow setup guide to build Power BI dashboard")
    print(f"   3. Create visualizations using the prepared datasets")
    print(f"   4. Share dashboard with stakeholders")

if __name__ == "__main__":
    # Create powerbi directory if it doesn't exist
    os.makedirs('powerbi', exist_ok=True)
    prepare_powerbi_data()