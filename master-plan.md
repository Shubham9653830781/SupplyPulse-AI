# SupplyPulse AI — Master Plan

> **Brand**: SupplyPulse AI  
> **Tagline**: Your Supply Chain Health Score, In Real Time  
> **Vision**: Unified health intelligence for global logistics — replacing isolated metrics with a single, actionable 0–100 score per shipment.

---

## 1. Problem Statement

Supply chains generate mountains of data — delay logs, weather feeds, port congestion reports, supplier scores, inventory levels, customs records, transport costs. Yet there is no single number that tells a stakeholder: **how healthy is this shipment right now?**

Existing solutions predict one thing (e.g., "will this be late?"). SupplyPulse AI compresses **7+ dimensions** into one interpretable Health Score with a reason + risk level.

---

## 2. Core Output

| Output | Format |
|---|---|
| **Health Score** | Integer 0–100 |
| **Reason** | Bullet list (green check / red cross) |
| **Risk Level** | Low / Medium / High / Critical |
| **Feature Importance** | SHAP waterfall per prediction |

### Example

```
Health Score = 82

Reason:
✓ Weather Normal
✓ Inventory Good
✗ Port Congestion High
✗ Supplier Delay Increasing

Risk: Medium
```

---

## 3. Input Dimensions (7 Pillars)

| # | Pillar | Source Features |
|---|---|---|
| 1 | Delivery Delay | ETA variation, actual vs scheduled days, late_delivery_risk |
| 2 | Weather | Weather condition severity, temperature, humidity, storm proximity |
| 3 | Port Congestion | Port congestion level, waiting time, loading/unloading time |
| 4 | Supplier Reliability | Supplier reliability score, lead time, historical defect rate |
| 5 | Transport Cost | Shipping cost, fuel price index, cost per unit weight |
| 6 | Customs Delay | Customs clearance time, customs hold flag |
| 7 | Inventory Level | Warehouse inventory level, stockout count, turnover ratio |

---

## 4. Datasets (Kaggle)

### Primary: Logistics & Supply Chain Dataset
- **Source**: `datasetengineer/logistics-and-supply-chain-dataset`
- **Records**: ~60,000 hourly rows (Jan 2021–Jan 2024)
- **Key columns**: Port congestion level, weather severity, warehouse inventory, order fulfillment, traffic congestion, shipping costs, ETA variation, supplier reliability, customs clearance time
- **Targets included**: Delay probability, disruption likelihood, risk classification
- **Size**: 15.5 MB

### Secondary: Global Supply Chain Risk & Logistics (2024–2026)
- **Source**: `nudratabbas/global-supply-chain-risk-and-logistics-2024-2026`
- **Records**: 5,000 shipments
- **Key columns**: Geopolitical risk score, fuel price index, carrier reliability, weather conditions, transport mode, distance, disruption flag
- **Size**: 485 KB

### Tertiary: DataCo SMART SUPPLY CHAIN
- **Source**: `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`
- **Records**: ~180,000 orders
- **Key columns**: Days for shipping (real/scheduled), delivery status, late_delivery_risk, benefit per order, sales per customer, product categories
- **Size**: 95.9 MB

### Dataset Combination Strategy

```
logistics_dataset.csv  ──┐
                        ├──► merged_df  (key: timestamp + route_id)
risk_logistics.csv     ──┘
                              │
dataco_supply_chain.csv  ────┤  (key: order_id / product_id)
                              ▼
                    Feature-engineered Health Score (label)
```

- Merge on time windows + geographic/route overlap
- Synthetic health score label derived from weighted sub-scores (see ML section)

---

## 5. Health Score Formula (Label Engineering)

Since no dataset has a native "health score", we derive one:

```
Health_Score = 100 - weighted_penalty
```

Where:

```
penalty =
    w1 * delay_risk_score       (0–100)
  + w2 * weather_penalty        (0–100)
  + w3 * port_congestion_score  (0–100)
  + w4 * supplier_risk_score    (0–100)
  + w5 * cost_efficiency_score  (0–100, inverted)
  + w6 * customs_risk_score     (0–100)
  + w7 * inventory_risk_score   (0–100)
```

- **Weights** initialised via domain heuristics, refined via regression on downstream outcomes
- Score is bucketed into: **Excellent (80–100)**, **Good (60–79)**, **Fair (40–59)**, **Poor (20–39)**, **Critical (0–19)**

---

## 6. ML Approach

### Models (Ensemble)

| Model | Role | Why |
|---|---|---|
| **XGBoost** | Primary regressor | Handles missing data, non-linear interactions, industry standard |
| **Random Forest** | Ensemble + SHAP baseline | Robust, interpretable, excellent for feature importance |
| **CatBoost** | Categorical-heavy variant | Handles categorical features (port, mode, supplier) natively |

No deep learning — the dataset is tabular (<100K rows), and explainability matters.

### Task Types

| Task | Target | Model |
|---|---|---|
| Regression | Health Score (0–100) | XGBoost, RF, CatBoost |
| Classification | Risk Level (4 classes) | Same models, softmax output |
| Multi-output | Score + Risk simultaneously | Multi-output wrapper |

### Evaluation

| Metric | Use |
|---|---|
| RMSE | Regression accuracy |
| MAE | Mean absolute score error |
| R² | Variance explained |
| F1 (macro) | Risk level classification |
| Cohen's Kappa | Ordinal classification quality |

---

## 7. SHAP Explainability

- Global SHAP: feature importance ranked across all shipments
- Local SHAP: per-shipment waterfall decomposition
- Displayed as: `✓ Weather Normal (+5 pts)` / `✗ Port Congestion High (-12 pts)`

---

## 8. Dashboard Features (Streamlit)

| Feature | Component |
|---|---|
| **Health Score Gauge** | Plotly gauge chart (0–100, colour-coded) |
| **Risk Meter** | Animated needle/dial |
| **Delay Probability** | Probability bar / gauge |
| **Feature Importance (SHAP)** | Horizontal bar chart + per-shipment waterfall |
| **Shipment Timeline** | Timeline of score over recent hours/days |
| **Supplier Ranking** | Table + bar: avg score per supplier |
| **Geographic Map** | Plotly scatter_mapbox: shipments colour-coded by score |
| **Filters** | Date range, supplier, port, transport mode, risk level |
| **Drill-down** | Click a shipment → full SHAP breakdown + raw features |

---

## 9. Work Breakdown

### Phase 1 — Data (Week 1)

- [x] Dataset research (done)
- [ ] Download 3 Kaggle datasets via API
- [ ] ETL pipeline (cleaning, merging, dedup)
- [ ] Feature engineering pipeline (lags, ratios, aggregates)
- [ ] Health Score label generation
- [ ] Train/validation/test split (70/15/15)

### Phase 2 — ML (Week 2)

- [ ] Baseline: Linear Regression + RF
- [ ] Primary: XGBoost hyperparameter tuning (Optuna)
- [ ] CatBoost with categorical features
- [ ] Multi-output (score + risk class)
- [ ] SHAP analysis (global + local)
- [ ] Model serialisation (joblib / pickle)

### Phase 3 — Dashboard (Week 3)

- [ ] Streamlit app scaffold
- [ ] Health Score gauge + risk meter
- [ ] SHAP waterfall integration
- [ ] Shipment timeline + map
- [ ] Supplier ranking page
- [ ] Filters and drill-down

### Phase 4 — Polish (Week 4)

- [ ] Branding (logo, colours, "SupplyPulse AI")
- [ ] Docker containerisation
- [ ] README + demo video
- [ ] Deploy (Streamlit Cloud / Hugging Face Spaces / EC2)
- [ ] Write blog post ("Building SupplyPulse AI")

---

## 10. Why This Stands Out

| What most people build | What SupplyPulse AI builds |
|---|---|
| Delivery time prediction | **Composite health score (7 dimensions)** |
| Binary delay classification | **Continuous 0–100 score + ordinal risk** |
| Single-model approach | **Ensemble (XGBoost + RF + CatBoost)** |
| Black-box prediction | **SHAP-explained score with reasons** |
| Static notebook | **Interactive Streamlit dashboard** |
| Generic naming | **Branded: SupplyPulse AI** |

---

## 11. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Datasets are synthetic | Use 3 datasets, cross-validate patterns; acknowledge in docs |
| Health score labels are engineered | Validate against real delay outcomes; weight sensitivity analysis |
| No real-time feed | Simulate with streaming via Streamlit caching + refresh |
| Class imbalance in risk levels | SMOTE + class weights in XGBoost |
