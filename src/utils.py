"""
SupplyPulse AI — Data Science Visualization & Charting Utilities
Provides clean, consistent, portfolio-grade Plotly and Matplotlib charts.
"""

from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Professional Supply-Chain Color Palette
COLOR_TEAL = "#00A896"
COLOR_CYAN = "#028090"
COLOR_NAVY = "#05668D"
COLOR_AMBER = "#F4A261"
COLOR_CORAL = "#E76F51"
COLOR_MINT = "#2A9D8F"
COLOR_SLATE = "#1E293B"
COLOR_LIGHT_BG = "#F8FAFC"
COLOR_DARK_TEXT = "#0F172A"

RISK_COLORS = {
    'Critical': '#EF4444',
    'High': '#F97316',
    'Medium': '#EAB308',
    'Good': '#3B82F6',
    'Excellent': '#10B981'
}


def create_health_gauge(score: float, title: str = "Shipment Health Score") -> go.Figure:
    """Renders an executive gauge indicator with 5 risk tier bands."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': COLOR_DARK_TEXT, 'family': 'sans-serif'}},
        number={'suffix': "/100", 'font': {'size': 36, 'color': COLOR_DARK_TEXT, 'weight': 600}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': COLOR_NAVY, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 30], 'color': '#FEE2E2'},    # Critical (soft red)
                {'range': [30, 50], 'color': '#FFEDD5'},   # High (soft orange)
                {'range': [50, 70], 'color': '#FEF9C3'},   # Medium (soft yellow)
                {'range': [70, 90], 'color': '#DBEAFE'},   # Good (soft blue)
                {'range': [90, 100], 'color': '#D1FAE5'}   # Excellent (soft green)
            ],
            'threshold': {
                'line': {'color': COLOR_CORAL, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': COLOR_DARK_TEXT}
    )
    return fig


def create_risk_distribution_chart(risk_counts: Dict[str, int]) -> go.Figure:
    """Renders a clean risk distribution bar chart ordered logically."""
    order = ['Critical', 'High', 'Medium', 'Good', 'Excellent']
    ordered_keys = [k for k in order if k in risk_counts]
    values = [risk_counts[k] for k in ordered_keys]
    colors = [RISK_COLORS.get(k, COLOR_CYAN) for k in ordered_keys]

    fig = go.Figure(data=[
        go.Bar(
            x=ordered_keys,
            y=values,
            marker_color=colors,
            text=[f"{v:,}" for v in values],
            textposition='auto',
            hovertemplate="<b>Risk Tier:</b> %{x}<br><b>Shipments:</b> %{y:,}<extra></extra>"
        )
    ])
    fig.update_layout(
        title="<b>Holdout Test Set — Risk Tier Distribution</b>",
        xaxis_title="Risk Classification Tier",
        yaxis_title="Shipment Volume",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=30, t=50, b=40)
    )
    return fig


def create_correlation_heatmap_fig(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Creates an interactive Plotly correlation heatmap for EDA."""
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="<b>Pearson Feature Correlation Matrix</b>"
    )
    fig.update_layout(
        height=550,
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(title="r")
    )
    return fig


def create_shap_bar_chart(sorted_shap: Any, max_display: int = 12) -> go.Figure:
    """Renders a horizontal bar chart of top absolute SHAP values."""
    if isinstance(sorted_shap, dict):
        if "sorted_features" in sorted_shap:
            sorted_shap = sorted_shap["sorted_features"]
        elif "shap_dict" in sorted_shap:
            sorted_shap = sorted(sorted_shap["shap_dict"].items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            sorted_shap = list(sorted_shap.items())

    top_items = list(sorted_shap[:max_display])
    top_items.reverse()  # ascending for horizontal bar chart

    features = [item[0] for item in top_items]
    values = [item[1] for item in top_items]
    colors = [COLOR_CORAL if v < 0 else COLOR_TEAL for v in values]

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            hovertemplate="<b>Feature:</b> %{y}<br><b>SHAP Attribution:</b> %{x:+.3f}<extra></extra>"
        )
    ])
    fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
    fig.update_layout(
        title=f"<b>Top {max_display} Feature Attributions (SHAP Impact)</b>",
        xaxis_title="SHAP Value (Contribution to Prediction)",
        yaxis_title="",
        template="plotly_white",
        height=400,
        margin=dict(l=180, r=30, t=50, b=40)
    )
    return fig


def create_scenario_comparison_chart(
    baseline_dict: Dict[str, Any],
    simulated_dict: Dict[str, Any]
) -> go.Figure:
    """Creates a side-by-side comparison for What-If scenario analysis."""
    metrics = ["Health Score", "ETA Deviation (hrs)", "Shipping Cost ($)", "Fuel Rate (L/hr)"]
    base_vals = [
        baseline_dict["health"],
        baseline_dict["eta"],
        baseline_dict["cost"],
        baseline_dict["fuel"]
    ]
    sim_vals = [
        simulated_dict["health"],
        simulated_dict["eta"],
        simulated_dict["cost"],
        simulated_dict["fuel"]
    ]

    fig = go.Figure(data=[
        go.Bar(name='Baseline', x=metrics, y=base_vals, marker_color=COLOR_SLATE),
        go.Bar(name='Simulated', x=metrics, y=sim_vals, marker_color=COLOR_TEAL)
    ])
    fig.update_layout(
        barmode='group',
        title="<b>Baseline vs Simulated Operational Outcome</b>",
        template="plotly_white",
        height=340,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
