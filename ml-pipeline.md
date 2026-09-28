# SupplyPulse AI — ML Pipeline

---

## 1. Pipeline Overview

```
Raw Data
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                               │
│  • Handle missing: median (numeric), mode (categorical)       │
│  • Outlier capping: IQR × 1.5                                 │
│  • Standardise timestamps → datetime + temporal features      │
│  • Encode categorical: target encoding + one-hot              │
│  • Scale numeric: StandardScaler (per model)                   │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                            │
│  • Lags: delay at t-1, t-7, t-30 (rolling windows)           │
│  • Ratios: cost_per_km, inventory_turnover                    │
│  • Interactions: port_congestion × weather, distance × mode   │
│  • Aggregates: mean delay per supplier, mean score per port   │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                  TRAIN / VAL / TEST SPLIT                      │
│  • Temporal split (time-based, not random)                    │
│  • Train: Jan 2021 – Jun 2023  (70%)                         │
│  • Val:   Jul 2023 – Jun 2024  (15%)                         │
│  • Test:  Jul 2024 – onward    (15%)                         │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │ XGBoost  │  │ Random   │  │ CatBoost │                    │
│  │ Regressor│  │ Forest   │  │ Regressor│                    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                    │
│       │              │              │                          │
│  Optuna tuning   Default tuned   Default tuned                 │
│  (100 trials)    (50 trials)     (50 trials)                   │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                   EVALUATION                                   │
│  • Regression: RMSE, MAE, R²                                 │
│  • Classification: Accuracy, F1-macro, Cohen Kappa           │
│  • Residual analysis (error vs feature values)               │
│  • Cross-validation consistency check                        │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                   ENSEMBLE SELECTION                           │
│  • Weighted average of top-2 models (val performance)         │
│  • Fallback: single best model if ensemble does not improve   │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                   SHAP EXPLAINABILITY                          │
│  • Global: summary_plot, bar_plot                             │
│  • Local: waterfall_plot per shipment                         │
│  • Export: shap_values saved with model                       │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                   SERIALISATION                                │
│  • Model: joblib dump                                         │
│  • Scaler: joblib dump                                        │
│  • SHAP explainer: pickle                                     │
│  • Feature list: JSON                                         │
│  • All → /models/ directory                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing Details

### 2.1 Missing Values

| Feature Type | Strategy | Implemnetation |
|---|---|---|
| Numeric (normal distrib.) | Median imputation | `SimpleImputer(strategy='median')` |
| Numeric (skewed) | Median imputation | Same; log transform after |
| Categorical | Mode imputation | `SimpleImputer(strategy='most_frequent')` |
| Target variables | Drop rows | ~2% missing — safe to drop |

### 2.2 Outlier Handling

```python
def cap_outliers(df, columns, factor=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = df[col].clip(lower, upper)
    return df
```

### 2.3 Encoding

| Encoder | Applied To | Notes |
|---|---|---|
| One-hot | transport_mode, product_category | Low cardinality (<10) |
| Target encoding | origin_port, destination_port, supplier_id | High cardinality, avoid leakage |
| Ordinal | risk_classification | Low → Moderate → High → Critical |

### 2.4 Scaling

- `StandardScaler` for tree-based models (optional — trees are scale-invariant but SHAP benefits)
- Applied consistently across train/val/test using train-fit transform

---

## 3. Feature Engineering

### 3.1 Temporal Features

| Feature | Derivation | Rationale |
|---|---|---|
| hour | timestamp.hour | Diurnal patterns |
| day_of_week | timestamp.dayofweek | Weekend effect |
| month | timestamp.month | Seasonality |
| quarter | timestamp.quarter | Quarterly trends |
| is_weekend | day_of_week >= 5 | Weekend operations |
| is_holiday | US federal holiday calendar | Reduced staff |

### 3.2 Rolling Windows

| Feature | Window | Aggregation |
|---|---|---|
| avg_delay_7d | 7 days | Mean of eta_variation |
| avg_delay_30d | 30 days | Mean of eta_variation |
| avg_congestion_7d | 7 days | Mean port_congestion |
| avg_supplier_reliability_30d | 30 days | Mean supplier score |
| total_cost_30d | 30 days | Sum shipping costs |

### 3.3 Interaction Features

| Feature | Formula | Rationale |
|---|---|---|
| congestion_weather | port_congestion × weather_severity | Compound disruption risk |
| distance_mode_cost | distance_km × shipping_costs / weight_mt | Cost efficiency normalised |
| supplier_delay_risk | (1 - supplier_reliability) × eta_variation | Supplier-driven delay |
| fuel_cost_burden | fuel_price_index × distance_km | Fuel impact on cost |

### 3.4 Aggregate Features

| Feature | Group By | Aggregation |
|---|---|---|
| supplier_avg_reliability | supplier_id | Mean |
| port_avg_congestion | port_id | Mean |
| route_avg_delay | origin + destination | Mean |
| mode_avg_cost | transport_mode | Mean |

---

## 4. Hyperparameter Tuning (Optuna)

### 4.1 XGBoost Search Space

```python
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    ...
    return cv_score
```

### 4.2 Random Forest Search Space

```python
def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
```

### 4.3 CatBoost Search Space

```python
def cb_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000, step=50),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
    }
```

**Trials**: 100 for XGBoost, 50 for RF/CatBoost  
**Pruning**: Optuna MedianPruner  
**Early stopping**: XGBoost / CatBoost use `early_stopping_rounds=50` on val set

---

## 5. Evaluation Protocol

### 5.1 Metrics

| Metric | Target | Threshold |
|---|---|---|
| RMSE | Health Score | < 10 (goal) |
| MAE | Health Score | < 7 (goal) |
| R² | Health Score | > 0.70 (goal) |
| F1 (macro) | Risk Level | > 0.75 (goal) |
| Cohen's Kappa | Risk Level | > 0.65 (goal) |

### 5.2 Baseline

Simple baselines to beat:

| Baseline | Method | Expected RMSE |
|---|---|---|
| Mean | Always predict mean score | ~18 |
| Median | Always predict median score | ~18 |
| Linear Regression | OLS on all features | ~12–14 |
| **XGBoost (tuned)** | Our target | **< 10** |

### 5.3 Validation Strategy

| Strategy | Reason |
|---|---|
| Temporal train/val/test split | Avoids time leakage |
| 5-fold cross-val on train (optional) | Check stability |
| Hold-out test set (never touched during tuning) | Unbiased final eval |

---

## 6. SHAP Integration

### 6.1 Global SHAP

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Bar plot (mean |SHAP|)
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

**Displayed as**: "Top 10 features driving health score"

### 6.2 Local SHAP (Per Shipment)

```python
# Single prediction
idx = 42
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx],
    feature_names=feature_names
))
```

**Displayed as**: Waterfall with `✓` / `✗` and point contributions:

```
Base Score: 65

Weather Severity    -8  ✗ Port Congestion
Port Congestion    -12  ✗ High
Inventory Level     +7  ✓ Good
Supplier Reliability +5  ✓ Reliable
...

Final Score: 82
```

---

## 7. Classification Head (Risk Level)

In addition to regression, we add a classification head:

```
Health Score  ──►  Bucketing ──► Risk Level

   0–19    →  Critical
  20–39    →  High
  40–59    →  Medium
  60–79    →  Good
  80–100   →  Excellent
```

### Multi-Output Approach

```python
from sklearn.multioutput import MultiOutputClassifier

# Option A: Separate classifier
clf = XGBClassifier(objective="multi:softmax", num_class=4)
clf.fit(X_train, y_risk_train)

# Option B: Multi-output (same model, two heads)
from sklearn.multioutput import MultiOutputRegressor
multi = MultiOutputRegressor(xgb_model)
multi.fit(X_train, y_multi_train)  # y_multi = [score, risk_encoded]
```

**Chosen**: Option A (separate classifier) — cleaner, easier to debug, and risk class benefits from ordinal loss.

---

## 8. Model Serialisation

```python
import joblib
import json
import pickle
import shap

# Save models
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(cb_model, "models/cb_model.pkl")
joblib.dump(ensemble_model, "models/ensemble_model.pkl")

# Save preprocessors
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

# Save SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
with open("models/shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# Save feature list
with open("models/feature_list.json", "w") as f:
    json.dump(list(feature_names), f)
```

---

## 9. Prediction Pipeline (Dashboard)

```python
def predict_health_score(features_dict):
    """
    features_dict: {
        "port_congestion_level": 7.2,
        "weather_condition_severity": 0.3,
        ...
    }
    Returns: {
        "score": 82,
        "risk": "Medium",
        "reasons": [...],
        "shap_waterfall": {...}
    }
    """
    df = pd.DataFrame([features_dict])
    df = preprocess(df, scaler, encoders)
    df = engineer_features(df)

    score = ensemble_model.predict(df)[0]
    risk = risk_classifier.predict(df)[0]

    shap_values = explainer.shap_values(df)
    reasons = format_reasons(shap_values, df.columns)

    return {
        "score": int(round(score)),
        "risk": risk_label(risk),
        "reasons": reasons,
        "shap_waterfall": shap_values.tolist(),
    }
```

---

## 10. Expected Performance

| Model | RMSE (val) | R² (val) | F1 (risk) | Training Time |
|---|---|---|---|---|
| Linear Regression (baseline) | ~13.5 | ~0.55 | ~0.58 | < 1 min |
| Random Forest (tuned) | ~9.8 | ~0.72 | ~0.74 | ~5 min |
| XGBoost (tuned) | ~8.5 | ~0.78 | ~0.78 | ~3 min |
| CatBoost (tuned) | ~8.8 | ~0.76 | ~0.77 | ~4 min |
| **Ensemble (XGB + CB)** | **~8.2** | **~0.80** | **~0.80** | — |

*Expected targets on the Logistics & Supply Chain dataset.*
