# 🚚 Logistics Anomaly Detection & Analytics Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Isolation_Forest-green.svg)
![Power BI](https://img.shields.io/badge/BI-Power_BI-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

An end-to-end machine learning system for detecting anomalies in global logistics operations, analyzing root causes, and visualizing insights through interactive Power BI dashboards.

## 📊 Project Overview

This project analyzes **32,065 international shipments** across 8 major carriers and 10 global routes to identify problematic patterns, predict delays, and quantify financial impacts. Using Isolation Forest machine learning, the system detects anomalous shipments with 10% precision and provides actionable insights for supply chain optimization.

### Key Achievements
- Detected **3,207 anomalous shipments** (10% of total)
- Identified **$34.27M in additional costs** across operations
- Analyzed **9.8% SLA breach rate** with root cause breakdown
- Built interactive Power BI dashboard with 5 analytical views
- Achieved 92.5% model precision in identifying problematic shipments

---

## 🎯 Business Impact

### Financial Insights
- **Total Cost Impact**: $34.27M in additional costs identified
- **Worst Performing Carrier**: CMA_CGM (12.2% anomaly rate)
- **Most Problematic Route**: Mumbai-Dubai (16.1% anomaly rate, $3.82M costs)
- **Average Delay**: 1,621 minutes (27 hours)
- **Peak Problem Period**: December 2024 (15.4% anomaly rate)

### Operational Findings
| Carrier | Anomaly Rate | Total Costs | Avg Cost/Shipment |
|---------|-------------|-------------|-------------------|
| **CMA_CGM** | 12.2% | $3.84M | $951.18 |
| **FedEx** | 11.9% | $4.74M | $1,173.32 |
| **MSC** | 11.2% | $4.29M | $1,077.80 |
| **DHL** | 10.6% | $4.00M | $1,036.14 |
| **UPS** | 7.5% | $4.27M | $1,049.86 |

---

## 🛠️ Technology Stack

### Machine Learning & Data Science
- **Python 3.10+**: Core programming language
- **scikit-learn**: Isolation Forest anomaly detection
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization

### Database & Storage
- **SQLite**: Optimized data storage and querying
- **SQL**: Complex analytical queries

### Business Intelligence
- **Power BI Desktop**: Interactive dashboard creation
- **DAX**: Custom measures and calculations

---

## 📁 Project Structure

```
logistics-anomaly-detection/
├── data/                                    # Data files
│   ├── logistics_base.csv                  # Original shipment data (32K rows)
│   ├── logistics_sla_rules.csv            # SLA definitions by carrier/route
│   ├── logistics_augmented.csv            # Enhanced with cost calculations
│   ├── logistics_features_engineered.csv  # ML-ready features (110+ columns)
│   ├── logistics_with_anomalies.csv       # Final predictions
│   └── logistics_analytics.db             # SQLite database (27MB)
│
├── scripts/                                 # Python analysis scripts
│   ├── 01_explore_data.py                 # Initial data exploration
│   ├── 02_setup_database.py               # Database creation
│   ├── 03_feature_engineering.py          # Feature creation (57 new features)
│   ├── 04_anomaly_detection.py            # Isolation Forest model
│   ├── 05_root_cause_analysis.py          # Root cause identification
│   └── 06_powerbi_preparation.py          # Dashboard data prep
│
├── models/                                  # Trained ML models
│   ├── isolation_forest_model.pkl         # Trained Isolation Forest
│   └── feature_scaler.pkl                 # StandardScaler for features
│
├── sql/                                     # SQL queries for Power BI
│   ├── dashboard_query.sql                # Main dashboard data
│   ├── carrier_performance.sql            # Carrier analytics
│   └── route_analysis.sql                 # Route performance
│
├── powerbi/                                 # Power BI files & data
│   ├── logistics_dashboard_data.csv       # Main dataset (32K rows)
│   ├── carrier_performance_summary.csv    # Carrier metrics (8 carriers)
│   ├── route_performance_summary.csv      # Route metrics (10 routes)
│   ├── monthly_trends.csv                 # Time series data
│   ├── risk_analysis.csv                  # Risk breakdown
│   ├── top_10_expensive_anomalies.csv     # Critical incidents
│   ├── top_10_longest_delays.csv          # Worst delays
│   ├── dashboard_metrics.csv              # KPI summary
│   ├── temporal.pbix                      # Power BI dashboard file
│   └── POWER_BI_SETUP_GUIDE.txt          # Setup instructions
│
├── outputs/                                 # Analysis outputs
│   ├── reports/                           # Text reports
│   │   ├── 01_data_exploration_summary.txt
│   │   ├── 02_database_schema.txt
│   │   ├── 03_modeling_features.csv
│   │   ├── 04_anomaly_analysis.txt
│   │   └── 05_root_cause_analysis.txt
│   └── visualizations/                    # Charts and graphs
│       ├── 04_anomaly_detection_results.png
│       └── 05_root_cause_analysis.png
│
└── README.md                               # This file
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
Power BI Desktop (for dashboard viewing)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Manikantachavvakula/Logistics_Anomaly.git
cd logistics-anomaly-detection
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy
```

3. **Run the analysis pipeline** (if recreating from scratch)
```bash
# Step 1: Explore data
python scripts/01_explore_data.py

# Step 2: Setup database
python scripts/02_setup_database.py

# Step 3: Engineer features
python scripts/03_feature_engineering.py

# Step 4: Detect anomalies
python scripts/04_anomaly_detection.py

# Step 5: Analyze root causes
python scripts/05_root_cause_analysis.py

# Step 6: Prepare Power BI data
python scripts/06_powerbi_preparation.py
```

4. **Open Power BI Dashboard**
```bash
# Open temporal.pbix in Power BI Desktop
# All data connections are pre-configured
```

---

## 🔬 Methodology

### 1. Data Engineering
- **Input**: 32,065 shipments with 54 base features
- **Output**: 110+ engineered features including:
  - Temporal patterns (month, quarter, day of week)
  - Carrier/route aggregations
  - Cost ratios and efficiency metrics
  - Deviation from historical averages
  - Interaction features

### 2. Anomaly Detection (Isolation Forest)
```python
IsolationForest(
    n_estimators=100,
    contamination=0.1,    # Expected 10% anomalies
    random_state=42
)
```

**Model Performance:**
- Detected: 3,207 anomalies (10.0% of dataset)
- Precision vs SLA Breaches: 92.5%
- Recall: Captured 9.8% actual SLA breaches

### 3. Root Cause Analysis
Statistical analysis identified key drivers:
- **Delay minutes**: Strongest cost correlation
- **SLA breach minutes**: Secondary driver
- **Route complexity**: Operational bottleneck
- **Customs delays**: 39.3% of total costs

---

## 📊 Power BI Dashboard

The dashboard includes 5 interactive pages:

### 1. Executive Summary
- **KPIs**: Total shipments (32K), anomalies (3,207), costs ($34.27M)
- **Carrier Performance**: Ranked by anomaly rate
- **Monthly Trends**: Anomaly rate over time

### 2. Carrier Analytics
- Carrier comparison by cost and anomaly rate
- Drill-through capability for detailed analysis
- Performance scorecards

### 3. Route Analytics
- Geographic route visualization
- Route performance heatmap
- Cost distribution by route

### 4. Anomaly Details
- Top 10 most expensive anomalies
- Risk category breakdown (High: 39%, Medium: 61%)
- Filterable anomaly table

### 5. Temporal Analysis
- Monthly/quarterly trends
- Day-of-week patterns
- Seasonal anomaly patterns

---

## 📈 Key Insights & Recommendations

### Immediate Actions
1. **Review CMA_CGM Contract** - Highest anomaly rate (12.2%)
2. **Optimize Mumbai-Dubai Route** - Most problematic (16.1% anomalies, $3.82M)
3. **Address December Bottleneck** - Peak problem month (15.4% anomaly rate)
4. **Implement Customs Fast-Track** - Customs delays account for 39.3% of costs

### Strategic Initiatives
- Deploy **early warning system** using anomaly scores
- Establish **carrier performance SLAs** with penalties
- Invest in **route optimization** for top 3 problematic routes
- Create **seasonal staffing plans** to handle December surge

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Shipments** | 32,065 |
| **Date Range** | Sept 2024 - Sept 2025 (1 year) |
| **Carriers** | 8 major global carriers |
| **Routes** | 10 international shipping lanes |
| **Features (Original)** | 54 columns |
| **Features (Engineered)** | 110+ columns |
| **Anomalies Detected** | 3,207 (10.0%) |
| **SLA Breaches** | 3,147 (9.8%) |
| **Total Additional Costs** | $34,271,734.98 |
| **Average Delay** | 1,621 minutes (27 hours) |
| **Database Size** | 27.14 MB |

---

## 🤖 Model Details

### Isolation Forest Parameters
- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Trees**: 100 estimators
- **Contamination**: 10% (expected anomaly rate)
- **Features Used**: 110 numerical features
- **Scaling**: StandardScaler normalization

### Top 10 Important Features
1. `delay_minutes`
2. `total_additional_cost`
3. `sla_breach_minutes`
4. `carrier_delay_minutes_mean`
5. `route_delay_minutes_mean`
6. `delay_vs_carrier_avg`
7. `cost_per_km`
8. `route_complexity`
9. `customs_delay_minutes`
10. `carrier_otd_history`

---

## 📚 Documentation

Detailed reports available in `outputs/reports/`:
- **01_data_exploration_summary.txt**: Initial data quality assessment
- **02_database_schema.txt**: Database structure and relationships
- **03_modeling_features.csv**: Complete feature list
- **04_anomaly_analysis.txt**: Model performance metrics
- **05_root_cause_analysis.txt**: Root cause breakdown with recommendations

---

## 🔄 Reproducibility

All analysis is **100% reproducible**:
1. Scripts are numbered sequentially (01-06)
2. Random seeds set (`random_state=42`)
3. Data pipeline fully automated
4. Models saved with `joblib`
5. SQL queries version-controlled

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- [ ] Add LSTM for time-series forecasting
- [ ] Implement real-time anomaly detection API
- [ ] Create automated email alerts for critical anomalies
- [ ] Add weather data integration
- [ ] Build Streamlit web interface

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Manik**
- GitHub:(https://github.com/Manikantachavvakula)
- Project: Logistics Anomaly Detection System
- Built: October 2024

---

## 🙏 Acknowledgments

- Dataset inspired by real-world logistics operations
- Isolation Forest implementation from scikit-learn
- Power BI visualizations created with Microsoft Power BI Desktop
- Built as part of an end-to-end ML project portfolio

---

## 📸 Key Results

### Cost Breakdown by Category
- **Customs Costs**: 39.3% of total ($13.47M)
- **Congestion Costs**: 19.8% of total ($6.80M)
- **Weather Costs**: 9.8% of total ($3.37M)
- **Disruption Costs**: 31.0% of total ($10.64M)

### Route Performance Summary
| Route | Anomaly Rate | Total Cost |
|-------|-------------|------------|
| Mumbai-Dubai | 16.1% | $3.82M |
| Hong_Kong-Tokyo | 11.2% | $3.08M |
| London-Singapore | 10.5% | $3.62M |
| New_York-Hamburg | 10.0% | $3.91M |

### Monthly Trend Highlights
- **Worst Month**: December 2024 (15.4% anomaly rate)
- **Best Month**: July 2025 (8.2% anomaly rate)
- **Average**: 10.0% anomaly rate across all months

---

**If you find this project useful, please star the repository!**

**Contact**: For questions or collaboration opportunities, feel free to reach out.