# SupplyPulse AI — Architecture

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  Kaggle CSV 1 ─┐                                                │
│  Kaggle CSV 2 ─┤──► ETL Pipeline ──► Merged DataFrame            │
│  Kaggle CSV 3 ─┘      (pandas)         (feather/parquet)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                          │
│  ┌────────────┐  ┌───────────┐  ┌──────────────────────────┐   │
│  │ Temporal   │  │ Aggregates │  │ Health Score Label Gen   │   │
│  │ (lags,     │  │ (rolling   │  │ (weighted 7-pillar       │   │
│  │  windows)  │  │  means)    │  │  penalty → 0–100)        │   │
│  └────────────┘  └───────────┘  └──────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ XGBoost  │  │ Random   │  │ CatBoost │  │ SHAP          │  │
│  │Regressor │  │ Forest   │  │ Regressor│  │ Explainer     │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│        │              │              │              │           │
│        └──────────────┴──────────────┴──────────────┘           │
│                         │                                       │
│                    [Ensemble Voting]                             │
│                         ▼                                       │
│                  ┌──────────────┐                                │
│                  │ Serialised   │                                │
│                  │ Model .pkl   │                                │
│                  └──────────────┘                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE / API                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  predict_health_score(features) → {score, risk, reasons} │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DASHBOARD (Streamlit)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ Score    │ │ Risk     │ │ SHAP     │ │ Timeline │ │ Map  │ │
│  │ Gauge    │ │ Meter    │ │Waterfall │ │ Chart    │ │      │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────┘ │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Filters: Date | Supplier | Port | Mode        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Data Layer

```
data/
├── raw/
│   ├── logistics_dataset.csv            # Primary: 60K rows, 26 cols
│   ├── global_supply_chain_risk.csv     # Secondary: 5K rows, 14 cols
│   └── dataco_supply_chain.csv          # Tertiary: 180K rows, 53 cols
├── processed/
│   ├── merged_dataset.feather           # Cleaned + merged (fast read)
│   └── engineered_features.feather      # After feature engineering
└── schema.yaml                          # Column definitions + types
```

**ETL Steps**:
1. Load CSVs with `pandas.read_csv()`
2. Standardise column names (snake_case, remove spaces)
3. Parse timestamps → datetime
4. Merge on temporal + geographic proximity
5. Handle missing values (median for numeric, mode for categorical)
6. Outlier clipping (IQR method)
7. Save as Feather (fast, compressed)

### 2.2 Feature Engineering

```
src/features/
├── build_features.py         # Master pipeline
├── temporal_features.py      # Hour of day, day of week, month, season
├── aggregate_features.py     # Rolling means (7-day, 30-day)
├── interaction_features.py   # port × weather, supplier × delay
├── target_engineering.py     # Health Score label derivation
└── feature_config.yaml       # Feature list + types
```

**Feature Groups**:

| Group | Example Features | Count |
|---|---|---|
| Raw numeric | port_congestion_level, weather_severity | ~12 |
| Temporal | hour_of_day, is_weekend, month, quarter | ~6 |
| Rolling | 7-day avg delay, 30-day avg cost | ~8 |
| Categorical | transport_mode, port_id, supplier_id | ~5 |
| Interaction | congestion × weather, distance × mode | ~4 |
| **Total** | | **~35** |

### 2.3 Model Layer

```
src/models/
├── train.py                   # Training orchestration
├── xgboost_model.py           # XGBoost regressor + classifier
├── random_forest_model.py     # RF regressor + classifier
├── catboost_model.py          # CatBoost regressor + classifier
├── ensemble.py                # Voting/averaging ensemble
├── evaluate.py                # Metrics (RMSE, MAE, R², F1)
├── tune.py                    # Optuna hyperparameter search
└── explain.py                 # SHAP explainer (global + local)
```

**Training Flow**:

```
split data → train models → tune hyperparams → evaluate → select best → explain
     │            │               │               │            │          │
     ▼            ▼               ▼               ▼            ▼          ▼
 70/15/15    3 models       Optuna 100      RMSE/MAE/    Ensemble   SHAP Tree
             in parallel    trials          F1/R²        or best    Explainer
```

### 2.4 Dashboard Layer

```
dashboard/
├── app.py                    # Streamlit entry point
├── pages/
│   ├── 01_Health_Score.py    # Score gauge + risk meter + reasons
│   ├── 02_Shipment_Timeline.py
│   ├── 03_Supplier_Ranking.py
│   ├── 04_Feature_Importance.py  # SHAP global + local
│   └── 05_Geographic_Map.py
├── components/
│   ├── score_gauge.py        # Plotly gauge chart component
│   ├── risk_meter.py         # Animated risk dial
│   ├── shap_waterfall.py     # SHAP waterfall renderer
│   └── filters.py            # Reusable filter sidebar
├── utils/
│   ├── model_loader.py       # Load serialised model + SHAP explainer
│   └── data_loader.py        # Load processed data + cache
└── assets/
    ├── logo.png
    └── style.css
```

**Streamlit Architecture**:

```
User opens app
       │
       ▼
  app.py (main, config, sidebar filters)
       │
       ├──► Page 1: Health Score (gauge + risk meter + SHAP reasons)
       ├──► Page 2: Timeline (score over time)
       ├──► Page 3: Supplier Ranking (avg score per supplier)
       ├──► Page 4: Feature Importance (global SHAP + local drill-down)
       └──► Page 5: Geographic Map (score-coded shipments)
```

### 2.5 Caching Strategy

```python
@st.cache_data(ttl=300)  # 5-minute cache
def load_data(): ...

@st.cache_resource
def load_model(): ...

@st.cache_data
def predict_batch(features_df): ...
```

---

## 3. Data Flow (End-to-End)

```
                      Model Training Flow
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐
│ Raw CSVs │───►│ ETL +     │───►│ Feature  │───►│ Model     │
│ (Kaggle) │    │ Clean     │    │ Engineer │    │ Train     │
└──────────┘    └───────────┘    └──────────┘    └───────────┘
                                                       │
                                                       ▼
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐
│ User     │◄───│ Streamlit │◄───│ Predict  │◄───│ Model.pkl │
│ Browser  │    │ Dashboard │    │ API      │    │ + SHAP    │
└──────────┘    └───────────┘    └──────────┘    └───────────┘
```

---

## 4. Deployment Architecture

```
┌──────────────────────────────────────────────────┐
│              Docker Container                     │
│  ┌────────────┐  ┌────────────┐                  │
│  │ Streamlit  │  │ Python     │                  │
│  │ (port 8501)│  │ Inference  │                  │
│  └────────────┘  │ Engine     │                  │
│                   └────────────┘                  │
│  ┌────────────┐  ┌────────────┐                  │
│  │ Model .pkl │  │ Data       │                  │
│  │ + SHAP     │  │ .feather   │                  │
│  └────────────┘  └────────────┘                  │
└──────────────────────────────────────────────────┘
         │
         ▼
Streamlit Cloud / Hugging Face Spaces / Railway

Resource: 1 vCPU, 2 GB RAM, 1 GB disk (fits entire pipeline)
```

---

## 5. Directory Structure (Final)

```
supplypulse-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── schema.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py           # Kaggle API download
│   │   ├── etl.py                 # Load, clean, merge
│   │   └── splitter.py            # Train/val/test split
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py      # Master pipeline
│   │   ├── temporal_features.py
│   │   ├── aggregate_features.py
│   │   ├── interaction_features.py
│   │   └── target_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   ├── catboost_model.py
│   │   ├── ensemble.py
│   │   ├── evaluate.py
│   │   ├── tune.py
│   │   └── explain.py
│   └── dashboard/
│       ├── app.py
│       ├── pages/
│       ├── components/
│       ├── utils/
│       └── assets/
├── models/
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── cb_model.pkl
│   ├── ensemble_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
├── requirements.txt
├── Dockerfile
├── README.md
├── master-plan.md
├── architecture.md
├── techstack.md
├── data-dictionary.md
└── ml-pipeline.md
```

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| No database | File-based (Feather) | Dataset fits in RAM; no DB ops needed |
| No real-time streaming | Simulated via cache refresh | MVP scope; real-time is Phase 5 |
| Ensemble over single model | Voting ensemble | Reliability; XGBoost + RF + CatBoost cover different patterns |
| SHAP over LIME | SHAP | Consistent, theoretically grounded, faster for tree models |
| Streamlit over Dash/Flask | Streamlit | Fastest path to interactive dashboard from Python |
| Feather over Parquet | Feather | Slightly faster read for same-machine use |
