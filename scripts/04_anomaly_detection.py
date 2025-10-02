"""
STEP 4: Anomaly Detection with Isolation Forest
This script builds our machine learning model to detect anomalous shipments.
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

print("="*80)
print("ANOMALY DETECTION - STEP 4")
print("="*80)

def detect_anomalies():
    # ============================================================================
    # 1. LOAD ENGINEERED FEATURES
    # ============================================================================
    print("\n📂 Loading engineered features...")
    
    # Load from database or CSV
    try:
        conn = sqlite3.connect('data/logistics_analytics.db')
        df = pd.read_sql_query("SELECT * FROM shipments_engineered", conn)
        conn.close()
        print(f"  ✅ Loaded {len(df):,} shipments from database")
    except:
        # Fallback to CSV
        df = pd.read_csv('data/logistics_features_engineered.csv')
        print(f"  ✅ Loaded {len(df):,} shipments from CSV")
    
    # ============================================================================
    # 2. SELECT FEATURES FOR MODELING
    # ============================================================================
    print("\n🔧 Selecting features for modeling...")
    
    # Define which features to use for anomaly detection
    # We'll exclude ID columns, dates, and target variables
    exclude_features = [
        'shipment_id', 'carrier_name', 'route', 'service_level',
        'shipment_date', 'planned_delivery', 'actual_delivery', 'timestamp'
    ]
    
    # Get all numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded features
    modeling_features = [f for f in numerical_features if f not in exclude_features]
    
    # Also remove target-like features to avoid data leakage
    target_like_features = ['is_anomaly', 'anomaly_score', 'risk_category']
    modeling_features = [f for f in modeling_features if f not in target_like_features]
    
    print(f"  ✅ Selected {len(modeling_features)} numerical features for modeling")
    print(f"  📊 First 10 features: {modeling_features[:10]}")
    
    # ============================================================================
    # 3. PREPARE DATA FOR MODELING
    # ============================================================================
    print("\n🔧 Preparing data for modeling...")
    
    # Select features for modeling
    X = df[modeling_features].copy()
    
    # Handle any missing values (just in case)
    X = X.fillna(X.median())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  ✅ Data shape: {X_scaled.shape}")
    print(f"  ✅ Features scaled and ready")
    
    # ============================================================================
    # 4. TRAIN ISOLATION FOREST MODEL
    # ============================================================================
    print("\n🤖 Training Isolation Forest model...")
    
    # Create and train the model
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,  # Expect about 10% anomalies (based on our SLA breach rate)
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the model
    iso_forest.fit(X_scaled)
    
    print("  ✅ Model trained successfully")
    
    # ============================================================================
    # 5. MAKE PREDICTIONS
    # ============================================================================
    print("\n🔮 Making anomaly predictions...")
    
    # Predict anomalies (-1 for anomalies, 1 for normal)
    predictions = iso_forest.predict(X_scaled)
    
    # Convert to binary (0 = normal, 1 = anomaly)
    df['is_anomaly'] = (predictions == -1).astype(int)
    
    # Get anomaly scores (the lower the score, the more anomalous)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    df['anomaly_score'] = anomaly_scores
    
    # Create risk categories based on scores
    df['risk_category'] = pd.cut(
        df['anomaly_score'],
        bins=[-float('inf'), -0.2, 0, 0.2, float('inf')],
        labels=['Critical', 'High', 'Medium', 'Low']
    )
    
    print(f"  📊 Anomalies detected: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")
    
    # ============================================================================
    # 6. ANALYZE RESULTS
    # ============================================================================
    print("\n📊 Analyzing detection results...")
    
    # Compare with known SLA breaches
    if 'sla_breach_minutes' in df.columns:
        sla_breaches = (df['sla_breach_minutes'] > 0)
        print(f"  📈 Actual SLA breaches: {sla_breaches.sum():,} ({sla_breaches.mean()*100:.1f}%)")
        
        # Check overlap between predicted anomalies and SLA breaches
        true_positives = ((df['is_anomaly'] == 1) & (sla_breaches == True)).sum()
        precision = true_positives / df['is_anomaly'].sum() if df['is_anomaly'].sum() > 0 else 0
        recall = true_positives / sla_breaches.sum() if sla_breaches.sum() > 0 else 0
        
        print(f"  🎯 Model Precision: {precision:.1%} (of predicted anomalies are real breaches)")
        print(f"  🎯 Model Recall: {recall:.1%} (of real breaches detected)")
    
    # Risk category breakdown
    print(f"\n  🚨 Risk Category Breakdown:")
    risk_counts = df['risk_category'].value_counts()
    for category, count in risk_counts.items():
        percentage = (count / len(df)) * 100
        print(f"     - {category}: {count:,} shipments ({percentage:.1f}%)")
    
    # ============================================================================
    # 7. FEATURE IMPORTANCE ANALYSIS
    # ============================================================================
    print("\n🔍 Analyzing feature importance...")
    
    # Get feature importance (average depth in trees)
    feature_importance = pd.DataFrame({
        'feature': modeling_features,
        'importance': np.mean([tree.feature_importances_ for tree in iso_forest.estimators_], axis=0)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print(f"\n  📈 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"     {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # ============================================================================
    # 8. SAVE MODEL AND RESULTS
    # ============================================================================
    print("\n💾 Saving model and results...")
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    # Save feature list used for modeling
    feature_usage = pd.DataFrame({
        'feature_name': modeling_features,
        'importance': feature_importance['importance'],
        'used_in_model': True
    })
    feature_usage.to_csv('outputs/reports/04_model_features_used.csv', index=False)
    
    print("  ✅ Saved model: models/isolation_forest_model.pkl")
    print("  ✅ Saved scaler: models/feature_scaler.pkl")
    print("  ✅ Saved feature list: outputs/reports/04_model_features_used.csv")
    
    # Save predictions to database
    conn = sqlite3.connect('data/logistics_analytics.db')
    df[['shipment_id', 'is_anomaly', 'anomaly_score', 'risk_category']].to_sql(
        'anomaly_predictions', conn, if_exists='replace', index=False
    )
    conn.close()
    print("  ✅ Saved predictions to database")
    
    # Save full results with predictions
    output_cols = ['shipment_id', 'carrier_name', 'route', 'shipment_date', 
                   'delay_minutes', 'sla_breach_minutes', 'total_additional_cost',
                   'is_anomaly', 'anomaly_score', 'risk_category']
    
    # Only include columns that exist in the dataframe
    available_cols = [col for col in output_cols if col in df.columns]
    df[available_cols].to_csv('data/logistics_with_anomalies.csv', index=False)
    print("  ✅ Saved results: data/logistics_with_anomalies.csv")
    
    # ============================================================================
    # 9. CREATE ANOMALY ANALYSIS REPORT
    # ============================================================================
    print("\n📋 Creating anomaly analysis report...")
    
    # Analyze anomalies by different dimensions
    anomaly_analysis = {}
    
    # By carrier
    if 'carrier_name' in df.columns:
        carrier_analysis = df.groupby('carrier_name').agg({
            'is_anomaly': ['count', 'sum', 'mean'],
            'anomaly_score': 'mean',
            'total_additional_cost': 'sum'
        }).round(4)
        carrier_analysis.columns = ['total_shipments', 'anomalies', 'anomaly_rate', 'avg_anomaly_score', 'total_cost']
        carrier_analysis['anomaly_rate'] = carrier_analysis['anomaly_rate'] * 100
        anomaly_analysis['by_carrier'] = carrier_analysis
    
    # By route
    if 'route' in df.columns:
        route_analysis = df.groupby('route').agg({
            'is_anomaly': ['count', 'sum', 'mean'],
            'anomaly_score': 'mean'
        }).round(4)
        route_analysis.columns = ['total_shipments', 'anomalies', 'anomaly_rate', 'avg_anomaly_score']
        route_analysis['anomaly_rate'] = route_analysis['anomaly_rate'] * 100
        anomaly_analysis['by_route'] = route_analysis
    
    # By risk category
    risk_analysis = df.groupby('risk_category').agg({
        'shipment_id': 'count',
        'delay_minutes': 'mean',
        'total_additional_cost': ['mean', 'sum'],
        'sla_breach_minutes': 'mean'
    }).round(2)
    risk_analysis.columns = ['count', 'avg_delay', 'avg_cost', 'total_cost', 'avg_breach_minutes']
    anomaly_analysis['by_risk'] = risk_analysis
    
    # Save analysis reports
    with open('outputs/reports/04_anomaly_analysis.txt', 'w') as f:
        f.write("ANOMALY DETECTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Shipments: {len(df):,}\n")
        f.write(f"Anomalies Detected: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)\n\n")
        
        if 'by_carrier' in anomaly_analysis:
            f.write("CARRIER ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(anomaly_analysis['by_carrier'].to_string() + "\n\n")
        
        f.write("RISK CATEGORY ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(anomaly_analysis['by_risk'].to_string() + "\n\n")
        
        f.write("TOP 10 FEATURES:\n")
        f.write("-" * 40 + "\n")
        for i, row in feature_importance.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}\n")
    
    print("  ✅ Analysis report saved: outputs/reports/04_anomaly_analysis.txt")
    
    # ============================================================================
    # 10. CREATE VISUALIZATIONS
    # ============================================================================
    print("\n📊 Creating visualizations...")
    
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    try:
        # Plot 1: Anomaly scores distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Anomaly Threshold')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Risk categories
        plt.subplot(2, 2, 2)
        risk_counts = df['risk_category'].value_counts()
        colors = ['#ff6b6b', '#ffa726', '#fff176', '#81c784']
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Shipment Risk Categories')
        
        # Plot 3: Anomalies by carrier (if available)
        plt.subplot(2, 2, 3)
        if 'carrier_name' in df.columns:
            carrier_anomalies = df.groupby('carrier_name')['is_anomaly'].mean().sort_values(ascending=False)
            carrier_anomalies.plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.title('Anomaly Rate by Carrier')
            plt.xlabel('Carrier')
            plt.ylabel('Anomaly Rate')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Carrier data not available', ha='center', va='center')
            plt.title('Anomaly Rate by Carrier')
        
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Feature importance
        plt.subplot(2, 2, 4)
        top_features = feature_importance.head(10).sort_values('importance', ascending=True)
        plt.barh(range(len(top_features)), top_features['importance'], color='lightseagreen')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        
        plt.savefig('outputs/visualizations/04_anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Visualizations saved: outputs/visualizations/04_anomaly_detection_results.png")
    except Exception as e:
        print(f"  ⚠️  Visualization failed: {e}")
    
    # ============================================================================
    # 11. FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("✨ ANOMALY DETECTION COMPLETE!")
    print("="*80)
    
    # Show top anomalies
    if 'total_additional_cost' in df.columns:
        top_anomalies = df[df['is_anomaly'] == 1].nlargest(5, 'total_additional_cost')
    else:
        top_anomalies = df[df['is_anomaly'] == 1].head(5)
    
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   • Total shipments analyzed: {len(df):,}")
    print(f"   • Anomalies detected: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"   • Critical risk shipments: {(df['risk_category'] == 'Critical').sum():,}")
    
    if 'total_additional_cost' in df.columns:
        anomaly_costs = df[df['is_anomaly'] == 1]['total_additional_cost'].sum()
        total_costs = df['total_additional_cost'].sum()
        print(f"   • Cost in anomalies: ${anomaly_costs:,.2f} ({anomaly_costs/total_costs*100:.1f}% of total costs)")
    
    print(f"\n🔍 TOP 5 ANOMALIES:")
    for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        cost = row.get('total_additional_cost', 'N/A')
        carrier = row.get('carrier_name', 'Unknown')
        route = row.get('route', 'Unknown')
        score = row.get('anomaly_score', 'N/A')
        print(f"   {i}. {carrier} - {route}: Score={score:.3f}, Cost=${cost if isinstance(cost, str) else cost:,.2f}")
    
    print(f"\n📌 Next Steps:")
    print(f"   1. Review anomaly analysis report")
    print(f"   2. Check visualizations")
    print(f"   3. Examine top anomalies in data/logistics_with_anomalies.csv")
    print(f"   4. Proceed to Root Cause Analysis (Step 5)")

if __name__ == "__main__":
    detect_anomalies()