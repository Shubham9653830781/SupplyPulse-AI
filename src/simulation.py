"""
SupplyPulse AI — What-If Scenario Simulation Engine
Evaluates operational changes by submitting baseline vs counterfactual parameters
through the actual XGBoost pipelines and calculating SHAP attribution deltas.
"""

import time
from typing import Dict, Any, List, Tuple
from src.prediction import engine
from src.explainability import explainer_engine


class SimulationEngine:
    """Orchestrates counterfactual simulation comparing baseline and modified features."""

    def simulate(
        self,
        baseline_features: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs real ML inference for both baseline and modified scenarios.
        Computes metric deltas and explainability shifts.
        """
        start_time = time.perf_counter()

        simulated_features = baseline_features.copy()
        simulated_features.update(overrides)

        # 1. Real Model Inferences
        base_preds = engine.predict_all(baseline_features)
        sim_preds = engine.predict_all(simulated_features)

        # 2. Metric comparisons
        # Health
        base_health = float(base_preds["health"]["raw_prediction"])
        sim_health = float(sim_preds["health"]["raw_prediction"])
        health_delta = round(sim_health - base_health, 2)

        # Risk
        base_risk = str(base_preds["risk_level"]["display_prediction"])
        sim_risk = str(sim_preds["risk_level"]["display_prediction"])

        # ETA
        base_eta = float(base_preds["eta"]["raw_prediction"])
        sim_eta = float(sim_preds["eta"]["raw_prediction"])
        eta_delta = round(sim_eta - base_eta, 2)

        # Cost
        base_cost = float(base_preds["cost"]["raw_prediction"])
        sim_cost = float(sim_preds["cost"]["raw_prediction"])
        cost_delta = round(sim_cost - base_cost, 2)

        # Carbon / Fuel
        base_fuel = float(base_preds["carbon"]["raw_prediction"])
        sim_fuel = float(sim_preds["carbon"]["raw_prediction"])
        fuel_delta = round(sim_fuel - base_fuel, 2)

        # Delay Flag
        base_delay = str(base_preds["delay"]["display_prediction"])
        sim_delay = str(sim_preds["delay"]["display_prediction"])

        # 3. SHAP Deltas (Health Score)
        base_shap_res = explainer_engine.explain_sample("health", baseline_features)
        sim_shap_res = explainer_engine.explain_sample("health", simulated_features)

        base_shap = base_shap_res["shap_dict"]
        sim_shap = sim_shap_res["shap_dict"]

        shap_deltas = {}
        for feature in sim_shap:
            b_val = base_shap.get(feature, 0.0)
            s_val = sim_shap.get(feature, 0.0)
            diff = s_val - b_val
            if abs(diff) > 0.01:
                shap_deltas[feature] = round(diff, 3)

        sorted_shap_deltas = sorted(shap_deltas.items(), key=lambda x: abs(x[1]), reverse=True)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        # 4. Synthesize plain-English interview explanation
        explanation = self._generate_simulation_explanation(
            overrides=overrides,
            health_delta=health_delta,
            cost_delta=cost_delta,
            eta_delta=eta_delta,
            base_risk=base_risk,
            sim_risk=sim_risk,
            sorted_shap_deltas=sorted_shap_deltas
        )

        return {
            "baseline": {
                "health": round(base_health, 2),
                "risk": base_risk,
                "eta": round(base_eta, 1),
                "cost": round(base_cost, 2),
                "fuel": round(base_fuel, 2),
                "delay": base_delay
            },
            "simulated": {
                "health": round(sim_health, 2),
                "risk": sim_risk,
                "eta": round(sim_eta, 1),
                "cost": round(sim_cost, 2),
                "fuel": round(sim_fuel, 2),
                "delay": sim_delay
            },
            "deltas": {
                "health": health_delta,
                "risk_changed": base_risk != sim_risk,
                "eta": eta_delta,
                "cost": cost_delta,
                "fuel": fuel_delta
            },
            "shap_deltas": sorted_shap_deltas,
            "execution_time_ms": elapsed_ms,
            "explanation": explanation
        }

    def _generate_simulation_explanation(
        self,
        overrides: Dict[str, Any],
        health_delta: float,
        cost_delta: float,
        eta_delta: float,
        base_risk: str,
        sim_risk: str,
        sorted_shap_deltas: List[Tuple[str, float]]
    ) -> List[str]:
        lines = []

        # Overrides summarized
        mod_strs = [f"**{k}** = `{v}`" for k, v in overrides.items()]
        lines.append(f"Simulated modifications: {', '.join(mod_strs)}.")

        # Health & Risk impact
        if health_delta > 0:
            lines.append(f"• **Health Score improved by +{health_delta:.2f} points**.")
        elif health_delta < 0:
            lines.append(f"• **Health Score dropped by {health_delta:.2f} points**.")
        else:
            lines.append("• **Health Score remained unchanged**.")

        if base_risk != sim_risk:
            lines.append(f"• **Risk Level shifted from {base_risk} → {sim_risk}**.")
        else:
            lines.append(f"• **Risk Level remained steady at {base_risk}**.")

        # Financial & ETA impact
        if cost_delta != 0:
            sign = "+" if cost_delta > 0 else ""
            lines.append(f"• **Transportation Cost Impact:** `{sign}${cost_delta:.2f}` per shipment.")
        if eta_delta != 0:
            sign = "+" if eta_delta > 0 else ""
            lines.append(f"• **Estimated Arrival Shift:** `{sign}{eta_delta:.1f}` hours.")

        # Top SHAP mechanism
        if sorted_shap_deltas:
            top_f, top_v = sorted_shap_deltas[0]
            sign = "+" if top_v > 0 else ""
            lines.append(f"• **Key Driver:** Changing `{top_f}` contributed `{sign}{top_v:.2f}` to the health score delta.")

        return lines


simulation_engine = SimulationEngine()
