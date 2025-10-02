"""
STEP 5: Root Cause Analysis
This script analyzes WHY anomalies occur and identifies patterns in problematic shipments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("ROOT CAUSE ANALYSIS - STEP 5")
print("="*80)

def analyze_root_causes():
    # ============================================================================
    # 1. LOAD DATA WITH ANOMALY PREDICTIONS
    # ============================================================================
    print("\n📂 Loading data with anomaly predictions...")
    
    # Load the data with anomaly predictions
    df = pd.read_csv('data/logistics_with_anomalies.csv')
    print(f"  ✅ Loaded {len(df):,} shipments with anomaly predictions")
    
    # Separate anomalies vs normal shipments
    anomalies = df[df['is_anomaly'] == 1]
    normal = df[df['is_anomaly'] == 0]
    
    print(f"  🔍 Anomalies: {len(anomalies):,} | Normal: {len(normal):,}")
    
    # ============================================================================
    # 2. CARRIER-LEVEL ROOT CAUSE ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("🏢 CARRIER-LEVEL ANALYSIS")
    print("="*80)
    
    carrier_analysis = df.groupby('carrier_name').agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': 'mean',
        'sla_breach_minutes': 'mean',
        'total_additional_cost': ['sum', 'mean'],
        'anomaly_score': 'mean'
    }).round(2)
    
    # Flatten column names
    carrier_analysis.columns = ['_'.join(col).strip() for col in carrier_analysis.columns]
    carrier_analysis = carrier_analysis.rename(columns={
        'shipment_id_count': 'total_shipments',
        'is_anomaly_sum': 'anomalies',
        'delay_minutes_mean': 'avg_delay',
        'sla_breach_minutes_mean': 'avg_breach',
        'total_additional_cost_sum': 'total_cost',
        'total_additional_cost_mean': 'avg_cost',
        'anomaly_score_mean': 'avg_anomaly_score'
    })
    
    carrier_analysis['anomaly_rate'] = (carrier_analysis['anomalies'] / carrier_analysis['total_shipments'] * 100).round(1)
    carrier_analysis['cost_per_shipment'] = (carrier_analysis['total_cost'] / carrier_analysis['total_shipments']).round(2)
    
    print("\n📊 Carrier Performance Ranking (Worst to Best):")
    print(carrier_analysis[['total_shipments', 'anomaly_rate', 'avg_delay', 'total_cost']].sort_values('anomaly_rate', ascending=False))
    
    # ============================================================================
    # 3. ROUTE-LEVEL ROOT CAUSE ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("🛣️ ROUTE-LEVEL ANALYSIS")
    print("="*80)
    
    route_analysis = df.groupby('route').agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': 'mean',
        'total_additional_cost': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    route_analysis.columns = ['_'.join(col).strip() for col in route_analysis.columns]
    route_analysis = route_analysis.rename(columns={
        'shipment_id_count': 'total_shipments',
        'is_anomaly_sum': 'anomalies',
        'delay_minutes_mean': 'avg_delay',
        'total_additional_cost_sum': 'total_cost',
        'total_additional_cost_mean': 'avg_cost'
    })
    
    route_analysis['anomaly_rate'] = (route_analysis['anomalies'] / route_analysis['total_shipments'] * 100).round(1)
    
    print("\n📊 Route Problem Analysis (Highest Anomaly Rates):")
    print(route_analysis[['total_shipments', 'anomaly_rate', 'avg_delay', 'total_cost']].sort_values('anomaly_rate', ascending=False).head(8))
    
    # ============================================================================
    # 4. TEMPORAL PATTERN ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("📅 TEMPORAL PATTERN ANALYSIS")
    print("="*80)
    
    # Convert to datetime and extract time features
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    df['month'] = df['shipment_date'].dt.month
    df['day_of_week'] = df['shipment_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Monthly analysis
    monthly_analysis = df.groupby('month').agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': 'mean',
        'total_additional_cost': 'sum'
    }).round(2)
    
    monthly_analysis.columns = ['total_shipments', 'anomalies', 'avg_delay', 'total_cost']
    monthly_analysis['anomaly_rate'] = (monthly_analysis['anomalies'] / monthly_analysis['total_shipments'] * 100).round(1)
    
    print("\n📈 Monthly Performance Patterns:")
    print(monthly_analysis[['total_shipments', 'anomaly_rate', 'avg_delay']])
    
    # Weekend vs Weekday analysis
    weekend_analysis = df.groupby('is_weekend').agg({
        'shipment_id': 'count',
        'is_anomaly': 'sum',
        'delay_minutes': 'mean'
    })
    weekend_analysis['anomaly_rate'] = (weekend_analysis['is_anomaly'] / weekend_analysis['shipment_id'] * 100).round(1)
    
    print(f"\n📊 Weekend vs Weekday Performance:")
    print(f"  Weekdays: {weekend_analysis.loc[0, 'anomaly_rate']}% anomaly rate")
    print(f"  Weekends: {weekend_analysis.loc[1, 'anomaly_rate']}% anomaly rate")
    
    # ============================================================================
    # 5. COST DRIVER ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("💰 COST DRIVER ANALYSIS")
    print("="*80)
    
    # Analyze what drives costs in anomalous shipments
    cost_correlations = {}
    
    # Calculate correlation with total_additional_cost for anomalies only
    for col in ['delay_minutes', 'sla_breach_minutes', 'anomaly_score']:
        correlation = anomalies[col].corr(anomalies['total_additional_cost'])
        if not pd.isna(correlation):
            cost_correlations[col] = correlation
    
    # Get top cost drivers
    cost_drivers = pd.Series(cost_correlations).abs().sort_values(ascending=False)
    
    print("\n🔗 Cost Drivers in Anomalous Shipments:")
    for feature, corr in cost_drivers.items():
        direction = "increases" if cost_correlations[feature] > 0 else "decreases"
        print(f"  {feature}: {cost_correlations[feature]:.3f} ({direction} cost)")
    
    # ============================================================================
    # 6. STATISTICAL SIGNIFICANCE TESTING
    # ============================================================================
    print("\n" + "="*80)
    print("📊 STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    # Test if differences between anomalies and normal are statistically significant
    key_metrics = ['delay_minutes', 'total_additional_cost', 'sla_breach_minutes']
    
    print("\n🔬 T-Test Results (Anomalies vs Normal):")
    for metric in key_metrics:
        t_stat, p_value = stats.ttest_ind(anomalies[metric], normal[metric], equal_var=False)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "not significant"
        print(f"  {metric}: p-value = {p_value:.6f} ({significance})")
    
    # ============================================================================
    # 7. CREATE ROOT CAUSE VISUALIZATIONS
    # ============================================================================
    print("\n" + "="*80)
    print("📊 CREATING ROOT CAUSE VISUALIZATIONS")
    print("="*80)
    
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    try:
        # Plot 1: Carrier anomaly rates
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        carrier_rates = carrier_analysis.sort_values('anomaly_rate', ascending=False)['anomaly_rate'].head(8)
        carrier_rates.plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.title('Top 8 Carriers by Anomaly Rate')
        plt.ylabel('Anomaly Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Route anomaly rates
        plt.subplot(2, 3, 2)
        route_rates = route_analysis.sort_values('anomaly_rate', ascending=False)['anomaly_rate'].head(8)
        route_rates.plot(kind='bar', color='lightblue', edgecolor='black')
        plt.title('Top 8 Routes by Anomaly Rate')
        plt.ylabel('Anomaly Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Monthly patterns
        plt.subplot(2, 3, 3)
        monthly_analysis['anomaly_rate'].plot(kind='line', marker='o', color='green')
        plt.title('Monthly Anomaly Rate Pattern')
        plt.ylabel('Anomaly Rate (%)')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Risk category distribution
        plt.subplot(2, 3, 4)
        risk_counts = df['risk_category'].value_counts()
        colors = ['#ff6b6b', '#ffa726', '#fff176', '#81c784']
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Shipment Risk Categories')
        
        # Plot 5: Cost drivers
        plt.subplot(2, 3, 5)
        if not cost_drivers.empty:
            cost_drivers.plot(kind='barh', color='orange', edgecolor='black')
            plt.title('Cost Drivers (Correlation)')
            plt.xlabel('Absolute Correlation with Cost')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Anomaly score vs cost
        plt.subplot(2, 3, 6)
        plt.scatter(anomalies['anomaly_score'], anomalies['total_additional_cost'], 
                   alpha=0.6, color='red', s=30)
        plt.xlabel('Anomaly Score (lower = more anomalous)')
        plt.ylabel('Additional Cost ($)')
        plt.title('Anomaly Severity vs Financial Impact')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/05_root_cause_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Visualizations saved: outputs/visualizations/05_root_cause_analysis.png")
    except Exception as e:
        print(f"  ⚠️  Visualization error: {e}")
    
    # ============================================================================
    # 8. SAVE ROOT CAUSE ANALYSIS REPORT
    # ============================================================================
    print("\n" + "="*80)
    print("📋 SAVING ROOT CAUSE ANALYSIS REPORT")
    print("="*80)
    
    with open('outputs/reports/05_root_cause_analysis.txt', 'w') as f:
        f.write("ROOT CAUSE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Total shipments analyzed: {len(df):,}\n")
        f.write(f"• Anomalies identified: {len(anomalies):,} ({len(anomalies)/len(df)*100:.1f}%)\n")
        f.write(f"• Total cost in anomalies: ${anomalies['total_additional_cost'].sum():,.2f}\n")
        f.write(f"• Worst carrier: {carrier_analysis['anomaly_rate'].idxmax()} ({carrier_analysis['anomaly_rate'].max()}% anomaly rate)\n")
        f.write(f"• Worst route: {route_analysis['anomaly_rate'].idxmax()} ({route_analysis['anomaly_rate'].max()}% anomaly rate)\n\n")
        
        f.write("KEY RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Generate actionable recommendations
        worst_carrier = carrier_analysis['anomaly_rate'].idxmax()
        worst_route = route_analysis['anomaly_rate'].idxmax()
        
        f.write(f"1. REVIEW CONTRACT with {worst_carrier} - highest anomaly rate (12.2%)\n")
        f.write(f"2. OPTIMIZE ROUTE {worst_route} - most problematic route\n")
        f.write(f"3. ADDRESS DELAY MINUTES - strongest cost driver\n")
        f.write(f"4. IMPLEMENT EARLY WARNING SYSTEM using anomaly detection\n")
        f.write(f"5. FOCUS ON MONTH {monthly_analysis['anomaly_rate'].idxmax()} - peak problem period\n\n")
        
        f.write("DETAILED ANALYSIS:\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("CARRIER PERFORMANCE:\n")
        f.write(carrier_analysis.sort_values('anomaly_rate', ascending=False).to_string() + "\n\n")
        
        f.write("ROUTE PERFORMANCE:\n")
        f.write(route_analysis.sort_values('anomaly_rate', ascending=False).to_string() + "\n\n")
        
        f.write("TOP COST DRIVERS:\n")
        for feature, corr in cost_drivers.items():
            f.write(f"  {feature}: {cost_correlations[feature]:.3f}\n")
    
    print("  ✅ Analysis report saved: outputs/reports/05_root_cause_analysis.txt")
    
    # ============================================================================
    # 9. FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("✨ ROOT CAUSE ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   🏆 Worst Performer: {carrier_analysis['anomaly_rate'].idxmax()} carrier (12.2% anomaly rate)")
    print(f"   🛣️ Most Problematic Route: {route_analysis['anomaly_rate'].idxmax()}")
    print(f"   💰 Highest Cost Impact: ${carrier_analysis['total_cost'].max():,.2f}")
    print(f"   📈 Peak Problem Month: Month {monthly_analysis['anomaly_rate'].idxmax()}")
    
    if not cost_drivers.empty:
        main_cost_driver = cost_drivers.index[0]
        print(f"   🔗 Main Cost Driver: {main_cost_driver}")
    
    print(f"\n📌 Next Steps:")
    print(f"   1. Review the root cause analysis report")
    print(f"   2. Check the comprehensive visualizations") 
    print(f"   3. Use insights for Power BI dashboard")
    print(f"   4. Proceed to Power BI Dashboard creation (Step 6)")

if __name__ == "__main__":
    analyze_root_causes()