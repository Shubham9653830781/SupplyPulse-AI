"""
SupplyPulse AI — Explainability Engine (SHAP)
Extracts local and global feature attributions using TreeExplainer on trained pipelines.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import shap
from src.prediction import engine, MODEL_REGISTRY


class ExplainabilityEngine:
    """Computes exact TreeExplainer SHAP values and interprets them for business and interviews."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExplainabilityEngine, cls).__new__(cls)
            cls._instance._init_explainers()
        return cls._instance

    def _init_explainers(self):
        self.explainers = {}
        for key, pipe in engine.pipelines.items():
            model = pipe.named_steps['model']
            self.explainers[key] = shap.TreeExplainer(model)

    def _get_ordered_features(self, model_key: str) -> List[str]:
        pipe = engine.pipelines[model_key]
        pre = pipe.named_steps['preprocessor']
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        return list(num_cols) + list(cat_cols)

    def explain_sample(
        self,
        model_key: str,
        features_dict: Dict[str, Any],
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature attributions for a single shipment.
        Handles multi-class classification (risk_level) and single-output models.
        """
        if model_key not in self.explainers:
            raise KeyError(f"No explainer found for model '{model_key}'.")

        pipe = engine.pipelines[model_key]
        explainer = self.explainers[model_key]
        pre = pipe.named_steps['preprocessor']
        meta = MODEL_REGISTRY[model_key]

        df_feat = engine._prepare_dataframe(model_key, features_dict)
        X_trans = pre.transform(df_feat)
        ordered_features = self._get_ordered_features(model_key)

        raw_shap = explainer.shap_values(X_trans)
        expected_val = explainer.expected_value

        # Handle multi-class (risk_level)
        if meta["type"] == "classifier" and len(np.shape(raw_shap)) == 3:
            # Shape is (1, 40, num_classes)
            if class_idx is None:
                # Default to predicted class
                pred_class_idx = int(pipe.predict(df_feat)[0])
                class_idx = pred_class_idx

            shap_vector = raw_shap[0, :, class_idx]
            if isinstance(expected_val, (list, tuple, np.ndarray)):
                base_value = float(expected_val[class_idx])
            else:
                base_value = float(expected_val)
        else:
            # Single output or binary
            if len(np.shape(raw_shap)) == 2:
                shap_vector = raw_shap[0, :]
            else:
                shap_vector = np.array(raw_shap).flatten()

            if isinstance(expected_val, (list, tuple, np.ndarray)):
                base_value = float(expected_val[0])
            else:
                base_value = float(expected_val)

        # Map to feature names
        shap_dict = {}
        for f, val in zip(ordered_features, shap_vector):
            shap_dict[f] = float(val)

        # Ranked attributions
        sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(f, v) for f, v in sorted_items if v > 0][:5]
        top_negative = [(f, v) for f, v in sorted_items if v < 0][:5]

        # Generate structured business interpretation
        pred_res = engine.predict(model_key, features_dict)
        display_pred = pred_res["display_prediction"]
        target_name = meta["name"]
        unit = meta["unit"]

        text_explanation = self._generate_interview_explanation(
            target_name=target_name,
            prediction=display_pred,
            unit=unit,
            base_value=base_value,
            top_positive=top_positive,
            top_negative=top_negative
        )

        return {
            "model_key": model_key,
            "target_name": target_name,
            "prediction": display_pred,
            "unit": unit,
            "base_value": round(base_value, 2),
            "shap_dict": shap_dict,
            "sorted_features": sorted_items,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "ordered_features": ordered_features,
            "raw_feature_values": {f: features_dict.get(f, "N/A") for f in ordered_features},
            "interview_explanation": text_explanation
        }

    def _generate_interview_explanation(
        self,
        target_name: str,
        prediction: Any,
        unit: str,
        base_value: float,
        top_positive: List[Tuple[str, float]],
        top_negative: List[Tuple[str, float]]
    ) -> List[str]:
        """Formats plain-English bullets for interview demonstration."""
        lines = [
            f"**Model Prediction:** {target_name} = **{prediction} {unit}** (Baseline Prior: {base_value:.2f})",
            "",
            "**Primary Contributing Factors (SHAP Attribution):**"
        ]

        # Top 3 positive
        for f, val in top_positive[:3]:
            lines.append(f"• **{f}** (`+{val:.2f}`): Pushed the predicted {target_name.lower()} upward.")

        # Top 3 negative
        for f, val in top_negative[:3]:
            lines.append(f"• **{f}** (`{val:.2f}`): Pulled the predicted {target_name.lower()} downward.")

        return lines


# Module singleton
explainer_engine = ExplainabilityEngine()
