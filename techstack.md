# SupplyPulse AI — Tech Stack

---

## 1. Language & Runtime

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Primary language |
| **Bun** | 1.x (optional) | For any JS helper scripts (Kaggle API wrap) |

---

## 2. Data Processing

| Library | Version | Purpose |
|---|---|---|
| **pandas** | 2.2.x | DataFrame operations, ETL, merging |
| **numpy** | 1.26.x | Numerical operations |
| **pyarrow** | 15.x | Feather file format (fast I/O) |
| **pandas-profiling / ydata-profiling** | 4.x | Exploratory data analysis reports |

---

## 3. Machine Learning

### Core Models

| Library | Version | Purpose |
|---|---|---|
| **xgboost** | 2.1.x | Primary gradient-boosted regressor |
| **scikit-learn** | 1.5.x | Random Forest, metrics, preprocessing, train/test split |
| **catboost** | 1.2.x | Categorical-feature-optimised regressor |

### Training Utilities

| Library | Version | Purpose |
|---|---|---|
| **optuna** | 3.6.x | Hyperparameter optimisation |
| **imbalanced-learn** | 0.12.x | SMOTE for risk class imbalance |
| **joblib** | 1.4.x | Model serialisation |
| **feature-engine** | 1.8.x | Advanced feature engineering (woe, etc.) |

### Explainability

| Library | Version | Purpose |
|---|---|---|
| **shap** | 0.45.x | SHAP tree explainer, waterfall, summary plots |

---

## 4. Dashboard & Visualisation

| Library | Version | Purpose |
|---|---|---|
| **streamlit** | 1.36.x | Web application framework |
| **plotly** | 5.24.x | Interactive charts (gauge, timeline, map, bars) |
| **matplotlib** | 3.9.x | Static SHAP plots, export figures |
| **seaborn** | 0.13.x | Statistical visualisations in notebooks |
| **streamlit-option-menu** | 0.4.x | Navigation sidebar |

---

## 5. Development & Quality

| Tool | Version | Purpose |
|---|---|---|
| **pytest** | 8.x | Unit tests (ETL, features, model) |
| **black** | 24.x | Code formatting |
| **ruff** | 0.5.x | Linting |
| **pre-commit** | 3.7.x | Pre-commit hooks |
| **notebook** | 7.x | Jupyter notebooks for exploration |

---

## 6. Infrastructure

| Tool | Version | Purpose |
|---|---|---|
| **Docker** | 24+ | Containerisation |
| **Git** | — | Version control |
| **Kaggle API** | 1.6.x | Automated dataset download |
| **pip / uv** | latest | Package management (prefer `uv` for speed) |

---

## 7. Deployment Targets

| Platform | Free Tier | Notes |
|---|---|---|
| **Streamlit Community Cloud** | Yes | Easiest; 1 GB RAM limit — sufficient |
| **Hugging Face Spaces** | Yes | Great for ML demos; built-in model hosting |
| **Railway** | Yes (limited) | 500 MB RAM free tier |
| **Render** | Yes | Web services with free tier |

---

## 8. Complete `requirements.txt`

```txt
# Core
pandas>=2.2.0
numpy>=1.26.0
pyarrow>=15.0.0

# ML
scikit-learn>=1.5.0
xgboost>=2.1.0
catboost>=1.2.0
optuna>=3.6.0
shap>=0.45.0
imbalanced-learn>=0.12.0
joblib>=1.4.0

# Dashboard
streamlit>=1.36.0
plotly>=5.24.0
matplotlib>=3.9.0
seaborn>=0.13.0

# Dev
pytest>=8.0.0
black>=24.0.0
ruff>=0.5.0
pre-commit>=3.7.0
jupyter>=7.0.0

# Data
kagglehub>=0.3.0
ydata-profiling>=4.0.0
```

---

## 9. Environment Variables

```bash
# Kaggle credentials (for dataset download)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key

# App config
STREAMLIT_SERVER_PORT=8501
MODEL_PATH=models/ensemble_model.pkl
DATA_PATH=data/processed/engineered_features.feather
```

---

## 10. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 11. Why This Stack?

| Requirement | Choice | Alternatives Considered & Rejected |
|---|---|---|
| Tabular ML | XGBoost + RF + CatBoost | TensorFlow/PyTorch (overkill, less interpretable) |
| Fast dashboard | Streamlit | Dash (more boilerplate), Gradio (less flexible for multi-page) |
| Explainability | SHAP | LIME (less stable), Eli5 (deprecated) |
| Hyperparameter tuning | Optuna | GridSearchCV (slow), Hyperopt (less modern API) |
| No deep learning | — | Deep learning not suitable for <100K tabular rows |

---

## 12. Installation

```bash
# Clone
git clone https://github.com/yourusername/supplypulse-ai.git
cd supplypulse-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# Install
pip install -r requirements.txt

# Download datasets
python src/data/download.py

# Run ETL
python src/data/etl.py

# Train models
python src/models/train.py

# Launch dashboard
streamlit run src/dashboard/app.py
```
