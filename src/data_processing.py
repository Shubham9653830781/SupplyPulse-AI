"""
SupplyPulse AI — Data Processing & Loading Module
Handles data ingestion, validation, preprocessing inspection, and summary statistics.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Candidate paths for the test dataset
CANDIDATE_DATA_PATHS = [
    os.path.join("data", "processed", "test.csv"),
    os.path.join("backend", "data", "preprocessed", "test.csv"),
    os.path.join("..", "data", "processed", "test.csv"),
    os.path.join("..", "backend", "data", "preprocessed", "test.csv")
]

# The 40 T0 features strictly allowed at order-placement time (leakage-free)
T0_NUMERICAL_FEATURES = [
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Category Id',
    'Customer Zipcode',
    'Department Id',
    'Latitude',
    'Longitude',
    'Order Item Discount',
    'Order Item Discount Rate',
    'Order Item Product Price',
    'Order Item Quantity',
    'Sales',
    'Order Item Total',
    'Product Category Id',
    'Product Price',
    'haversine_distance_km',
    'order_month',
    'order_day_of_week',
    'is_weekend',
    'order_month_sin',
    'order_month_cos',
    'order_day_sin',
    'order_day_cos'
]

T0_CATEGORICAL_FEATURES = [
    'payment_type',
    'Category Name',
    'destination_city',
    'destination_country',
    'Customer Segment',
    'Customer State',
    'Customer Street',
    'Department Name',
    'Market',
    'origin_city',
    'origin_country',
    'order date (DateOrders)',
    'Order Region',
    'Order State',
    'product_name',
    'Shipping Mode'
]

ALL_T0_FEATURES = T0_NUMERICAL_FEATURES + T0_CATEGORICAL_FEATURES

# The 8 target variables
TARGET_VARIABLES = {
    'health_score': {
        'name': 'Health Score',
        'type': 'Regression',
        'range': '0 - 100',
        'unit': 'Score',
        'description': 'Composite operational health score combining delay, disruption, and logistics risk'
    },
    'risk_classification': {
        'name': 'Risk Classification',
        'type': 'Multi-class Classification',
        'classes': ['Critical', 'High', 'Medium', 'Good', 'Excellent'],
        'unit': 'Category',
        'description': 'Categorical risk tier bucketed from operational health score'
    },
    'is_delayed': {
        'name': 'Delay Flag',
        'type': 'Binary Classification',
        'classes': ['0 (On-Time)', '1 (Delayed)'],
        'unit': 'Flag',
        'description': 'Indicator whether shipment arrived past the scheduled delivery window'
    },
    'eta_variation_hours': {
        'name': 'ETA Variation',
        'type': 'Regression',
        'range': '-48 to +96 hrs',
        'unit': 'Hours',
        'description': 'Variance between actual delivery hours and scheduled delivery hours'
    },
    'supplier_reliability_score': {
        'name': 'Supplier Reliability',
        'type': 'Regression',
        'range': '0 - 100',
        'unit': 'Score',
        'description': 'Historical fulfillment performance score for the supplier/fulfillment channel'
    },
    'shipping_costs': {
        'name': 'Shipping Cost',
        'type': 'Regression',
        'range': '$0.66 - $291.00',
        'unit': 'USD ($)',
        'description': 'Transportation and delivery charges based on order volume and shipping mode'
    },
    'fuel_consumption_rate': {
        'name': 'Fuel / Carbon Rate',
        'type': 'Regression',
        'range': '0.09 - 23.73',
        'unit': 'L/hr',
        'description': 'Estimated fuel consumption and carbon footprint proxy for the transit route'
    },
    'disruption_occurred': {
        'name': 'Disruption Occurred',
        'type': 'Binary Classification',
        'classes': ['0 (Normal / Delivered)', '1 (Disrupted / Canceled)'],
        'unit': 'Flag',
        'description': 'Severe supply chain bottleneck resulting in cancellation or route breakdown'
    }
}


def find_data_file() -> str:
    """Locates the test CSV file across possible working directories."""
    for path in CANDIDATE_DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not locate test dataset. Checked paths: {CANDIDATE_DATA_PATHS}"
    )


def load_dataset() -> pd.DataFrame:
    """
    Loads the real holdout test dataset.
    Standardizes column types and extracts date features if needed.
    """
    path = find_data_file()
    df = pd.read_csv(path)
    
    # Ensure created_at / order_date is parsed as datetime
    if 'order date (DateOrders)' in df.columns:
        df['order_datetime'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    
    # Standardize carrier and supplier references if needed
    if 'carrier' not in df.columns and 'Shipping Mode' in df.columns:
        df['carrier'] = df['Shipping Mode']
    if 'supplier' not in df.columns:
        df['supplier'] = "DataCo Global Supply"
        
    return df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes summary dimensions and high-level KPIs."""
    total_records = len(df)
    total_columns = len(df.columns)
    
    missing_count = int(df[ALL_T0_FEATURES].isnull().sum().sum())
    duplicate_count = int(df.duplicated(subset=['shipment_id']).sum()) if 'shipment_id' in df.columns else int(df.duplicated().sum())
    
    avg_health = float(df['health_score'].mean()) if 'health_score' in df.columns else 0.0
    delay_rate = float((df['is_delayed'] == 1).mean() * 100) if 'is_delayed' in df.columns else 0.0
    disruption_rate = float((df['disruption_occurred'] == 1).mean() * 100) if 'disruption_occurred' in df.columns else 0.0
    avg_cost = float(df['shipping_costs'].mean()) if 'shipping_costs' in df.columns else 0.0
    avg_scheduled_days = float(df['Days for shipment (scheduled)'].mean()) if 'Days for shipment (scheduled)' in df.columns else 0.0
    avg_eta_var = float(df['eta_variation_hours'].mean()) if 'eta_variation_hours' in df.columns else 0.0
    
    risk_distribution = df['risk_classification'].value_counts().to_dict() if 'risk_classification' in df.columns else {}
    mode_distribution = df['Shipping Mode'].value_counts().to_dict() if 'Shipping Mode' in df.columns else {}
    segment_distribution = df['Customer Segment'].value_counts().to_dict() if 'Customer Segment' in df.columns else {}
    
    return {
        'total_records': total_records,
        'total_columns': total_columns,
        't0_feature_count': len(ALL_T0_FEATURES),
        'missing_values': missing_count,
        'duplicate_records': duplicate_count,
        'avg_health_score': round(avg_health, 2),
        'delay_rate_pct': round(delay_rate, 2),
        'disruption_rate_pct': round(disruption_rate, 2),
        'avg_shipping_cost_usd': round(avg_cost, 2),
        'avg_scheduled_days': round(avg_scheduled_days, 1),
        'avg_eta_variation_hours': round(avg_eta_var, 1),
        'risk_distribution': risk_distribution,
        'mode_distribution': mode_distribution,
        'segment_distribution': segment_distribution
    }


def get_sample_shipment(df: pd.DataFrame, index: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
    """Retrieves a specific or random shipment record formatted for prediction."""
    if index is None or index < 0 or index >= len(df):
        index = int(np.random.randint(0, len(df)))
    row = df.iloc[index]
    return index, row.to_dict()
