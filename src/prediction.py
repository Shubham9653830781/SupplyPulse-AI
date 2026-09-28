"""
SupplyPulse AI — Prediction Engine
Loads the 8 trained Scikit-Learn XGBoost pipelines and executes verified inference.
"""

import os
import json
import joblib
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

CANDIDATE_MODEL_DIRS = [
    os.path.join("models"),
    os.path.join("backend", "ml", "artifacts"),
    os.path.join("..", "models"),
    os.path.join("..", "backend", "ml", "artifacts")
]

MODEL_REGISTRY = {
    "health": {
        "file": "xgb_regressor.pkl",
        "target_col": "health_score",
        "name": "Shipment Health Score",
        "type": "regressor",
        "unit": "points (0-100)"
    },
    "risk_level": {
        "file": "xgb_classifier.pkl",
        "target_col": "risk_classification",
        "name": "Risk Classification",
        "type": "classifier",
        "label_encoder": "xgb_classifier_label_encoder.pkl",
        "unit": "Tier"
    },
    "delay": {
        "file": "delay_xgb.pkl",
        "target_col": "is_delayed",
        "name": "Late Delivery Risk",
        "type": "binary_classifier",
        "labels": {0: "On-Time", 1: "Delayed"},
        "unit": "Status"
    },
    "eta": {
        "file": "eta_xgb.pkl",
        "target_col": "eta_variation_hours",
        "name": "ETA Deviation",
        "type": "regressor",
        "unit": "hours"
    },
    "supplier": {
        "file": "supplier_xgb.pkl",
        "target_col": "supplier_reliability_score",
        "name": "Supplier Reliability",
        "type": "regressor",
        "unit": "points (0-100)"
    },
    "cost": {
        "file": "cost_xgb.pkl",
        "target_col": "shipping_costs",
        "name": "Transportation Cost",
        "type": "regressor",
        "unit": "USD ($)"
    },
    "carbon": {
        "file": "carbon_xgb.pkl",
        "target_col": "fuel_consumption_rate",
        "name": "Fuel / Carbon Burn Rate",
        "type": "regressor",
        "unit": "L/hr"
    },
    "inventory": {
        "file": "inventory_xgb.pkl",
        "target_col": "disruption_occurred",
        "name": "Disruption Likelihood",
        "type": "binary_classifier",
        "labels": {0: "Normal Flow", 1: "Route Disrupted"},
        "unit": "Status"
    }
}


class ModelEngine:
    """Manages model artifacts, preprocessing pipelines, and inference."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelEngine, cls).__new__(cls)
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        self.artifacts_dir = self._locate_artifacts_dir()
        self.pipelines: Dict[str, Any] = {}
        self.features_lists: Dict[str, List[str]] = {}
        self.label_encoders: Dict[str, Any] = {}
        self.experiments_history: List[Dict[str, Any]] = []
        self.is_loaded = False
        self._load_artifacts()

    def _locate_artifacts_dir(self) -> str:
        for d in CANDIDATE_MODEL_DIRS:
            if os.path.exists(os.path.join(d, "xgb_regressor.pkl")):
                return d
        raise FileNotFoundError(f"Could not find model artifacts in candidate paths: {CANDIDATE_MODEL_DIRS}")

    def _load_artifacts(self):
        if self.is_loaded:
            return

        # 1. Load pipelines & feature lists
        for model_key, meta in MODEL_REGISTRY.items():
            model_path = os.path.join(self.artifacts_dir, meta["file"])
            json_name = 'feature_names.json' if meta["file"] in ['xgb_regressor.pkl', 'xgb_classifier.pkl'] else meta["file"].replace('.pkl', '_features.json')
            feature_path = os.path.join(self.artifacts_dir, json_name)

            if os.path.exists(model_path) and os.path.exists(feature_path):
                self.pipelines[model_key] = joblib.load(model_path)
                with open(feature_path, "r") as f:
                    self.features_lists[model_key] = json.load(f)

            if "label_encoder" in meta:
                le_path = os.path.join(self.artifacts_dir, meta["label_encoder"])
                if os.path.exists(le_path):
                    self.label_encoders[model_key] = joblib.load(le_path)

        # 2. Load experiment tracking history (real metrics and hyperparameters)
        exp_candidates = [
            os.path.join(self.artifacts_dir, "experiments_history.json"),
            os.path.join("models", "experiments_history.json"),
            os.path.join("backend", "ml", "experiments_history.json"),
        ]
        for exp_path in exp_candidates:
            if os.path.exists(exp_path):
                try:
                    with open(exp_path, "r") as f:
                        self.experiments_history = json.load(f)
                    break
                except Exception:
                    pass

        self.is_loaded = True

    def get_experiment_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Maps experiment history to models by target variable."""
        metrics_by_target = {}
        for exp in self.experiments_history:
            target = exp.get("target")
            if target and target not in metrics_by_target:
                metrics_by_target[target] = exp
        return metrics_by_target

    def _prepare_dataframe(self, model_key: str, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """Aligns input dict to the expected feature schema and types."""
        required = self.features_lists.get(model_key, [])
        aligned = {}
        for f in required:
            val = features_dict.get(f, 0)
            aligned[f] = [val]

        df = pd.DataFrame(aligned)
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype('category')
        return df

    def predict(self, model_key: str, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Executes prediction for a specific model key."""
        if not self.is_loaded:
            self._load_artifacts()

        if model_key not in self.pipelines:
            raise KeyError(f"Model '{model_key}' not loaded.")

        pipeline = self.pipelines[model_key]
        meta = MODEL_REGISTRY[model_key]
        df_feat = self._prepare_dataframe(model_key, features_dict)

        raw_pred = pipeline.predict(df_feat)[0]
        confidence = 100.0
        probabilities = None
        display_prediction = raw_pred

        if meta["type"] == "classifier":
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(df_feat)[0]
                confidence = float(max(probs) * 100)
                model_step = pipeline.named_steps['model']
                classes = model_step.classes_
                probabilities = {str(c): float(p) for c, p in zip(classes, probs)}

            if model_key in self.label_encoders:
                le = self.label_encoders[model_key]
                display_prediction = str(le.inverse_transform([int(raw_pred)])[0])
            else:
                display_prediction = str(raw_pred)

        elif meta["type"] == "binary_classifier":
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(df_feat)[0]
                confidence = float(max(probs) * 100)
                probabilities = {"0": float(probs[0]), "1": float(probs[1])}
            label_map = meta.get("labels", {})
            display_prediction = label_map.get(int(raw_pred), str(raw_pred))

        else:
            # Regressor
            display_prediction = round(float(raw_pred), 2)

        return {
            "model_key": model_key,
            "target": meta["target_col"],
            "name": meta["name"],
            "raw_prediction": raw_pred,
            "display_prediction": display_prediction,
            "unit": meta["unit"],
            "confidence": round(confidence, 1),
            "probabilities": probabilities
        }

    def predict_all(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs concurrent predictions across all 8 models for a single shipment record."""
        results = {}
        for key in MODEL_REGISTRY.keys():
            results[key] = self.predict(key, features_dict)
        return results


# Module singleton
engine = ModelEngine()
