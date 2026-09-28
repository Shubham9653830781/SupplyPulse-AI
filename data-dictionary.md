# SupplyPulse AI — Data Dictionary

---

## 1. Dataset Sources

| # | Dataset | Kaggle ID | Rows | Cols | Size | License |
|---|---|---|---|---|---|---|
| 1 | Logistics & Supply Chain Dataset | `datasetengineer/logistics-and-supply-chain-dataset` | ~60,000 | 26 | 15.5 MB | CC0 |
| 2 | Global Supply Chain Risk & Logistics (2024-2026) | `nudratabbas/global-supply-chain-risk-and-logistics-2024-2026` | 5,000 | 14 | 485 KB | CC0 |
| 3 | DataCo SMART SUPPLY CHAIN | `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis` | ~180,000 | 53 | 95.9 MB | CC0 |

---

## 2. Primary Dataset: Logistics & Supply Chain

**File**: `logistics_and_supply_chain_dataset.csv`  
**Rows**: ~60,000 | **Cols**: 26 | **Period**: Jan 2021 – Jan 2024

### Columns

| # | Column | Type | Description | Use in Score |
|---|---|---|---|---|
| 1 | timestamp | datetime | Hourly recording timestamp | Temporal features |
| 2 | vehicle_gps_latitude | float | GPS lat of vehicle | Map visualisation |
| 3 | vehicle_gps_longitude | float | GPS lon of vehicle | Map visualisation |
| 4 | fuel_consumption_rate | float | Fuel use (L/hr) | Cost efficiency |
| 5 | eta_variation_hours | float | ETA - Actual (hours) | **Delivery Delay** |
| 6 | traffic_congestion_level | float | 0–10 scale | Port Congestion |
| 7 | warehouse_inventory_level | float | Current warehouse units | **Inventory Level** |
| 8 | loading_unloading_time | float | Hours spent loading/unloading | Port Congestion |
| 9 | handling_equipment_availability | int | 0=unavailable, 1=available | Supplier Reliability |
| 10 | order_fulfillment_status | int | 0=not fulfilled, 1=fulfilled | **Delivery Delay** |
| 11 | weather_condition_severity | float | 0–1 scale | **Weather** |
| 12 | port_congestion_level | float | 0–10 scale | **Port Congestion** |
| 13 | shipping_costs | float | USD | **Transport Cost** |
| 14 | supplier_reliability_score | float | 0–1 scale | **Supplier Reliability** |
| 15 | lead_time_days | float | Supplier delivery time | Supplier Reliability |
| 16 | historical_demand | float | Units demanded | Inventory Level |
| 17 | iot_temperature | float | °C from IoT sensors | Weather |
| 18 | cargo_condition_status | int | 0=poor, 1=good | Delivery Delay |
| 19 | route_risk_level | float | 0–10 scale | Composite risk |
| 20 | customs_clearance_time | float | Hours in customs | **Customs Delay** |
| 21 | driver_behavior_score | float | 0–1 scale | Supplier Reliability |
| 22 | fatigue_monitoring_score | float | 0–1 scale | Supplier Reliability |
| 23 | disruption_likelihood_score | float | 0–1 (target) | Validation |
| 24 | delay_probability | float | 0–1 (target) | Validation |
| 25 | risk_classification | str | Low/Moderate/High (target) | Validation |
| 26 | delivery_time_deviation | float | Hours deviation (target) | Validation |

### Target Variables Available

| Target | Type | Use |
|---|---|---|
| disruption_likelihood_score | regression (0–1) | Validate health score |
| delay_probability | regression (0–1) | Validate health score |
| risk_classification | classification (3-class) | Validate risk level |
| delivery_time_deviation | regression (hours) | Validate delay component |

---

## 3. Secondary Dataset: Global Supply Chain Risk

**File**: `global_supply_chain_risk_2026.csv`  
**Rows**: 5,000 | **Cols**: 14 | **Period**: 2024–2026

### Columns

| # | Column | Type | Description | Use |
|---|---|---|---|---|
| 1 | shipment_id | str | Unique identifier | Key |
| 2 | date | datetime | Dispatch date | Temporal |
| 3 | origin_port | str | Origin hub | Categorical |
| 4 | destination_port | str | Destination hub | Categorical |
| 5 | transport_mode | str | Sea/Air/Rail/Road | Categorical |
| 6 | product_category | str | Electronics, Pharma, etc. | Categorical |
| 7 | distance_km | float | Total distance | Cost / Delay |
| 8 | weight_mt | float | Metric tons | **Transport Cost** |
| 9 | fuel_price_index | float | Normalised fuel cost | **Transport Cost** |
| 10 | geopolitical_risk_score | float | 0–10 (0=stable) | **Supplier Reliability** |
| 11 | weather_conditions | str | Clear/Stormy/Rainy/Fog | **Weather** |
| 12 | carrier_reliability | float | 0–100% | **Supplier Reliability** |
| 13 | lead_time_days | float | Total transit days | **Delivery Delay** |
| 14 | disruption_occurred | int | 0/1 target | Validation |

---

## 4. Tertiary Dataset: DataCo SMART SUPPLY CHAIN

**File**: `DataCoSupplyChainDataset.csv`  
**Rows**: ~180,000 | **Cols**: 53 | **Period**: Multi-year

### Key Columns for Our Use

| # | Column | Type | Description | Use |
|---|---|---|---|---|
| 1 | Type | str | DEBIT/TRANSFER | Payment type |
| 2 | Days for shipping (real) | int | Actual shipping days | **Delivery Delay** |
| 3 | Days for shipment (scheduled) | int | Planned shipping days | **Delivery Delay** |
| 4 | Benefit per order | float | Profit | Cost efficiency |
| 5 | Sales per customer | float | Revenue | Cost efficiency |
| 6 | Delivery Status | str | Late delivery status | Validation |
| 7 | Late_delivery_risk | int | 0/1 flag | **Delivery Delay** |
| 8 | Category Name | str | Product category | Categorical |
| 9 | Customer City / State / Country | str | Geolocation | Map |
| 10 | Order City / State / Country | str | Geolocation | Map |
| 11 | Order Region | str | Region | Categorical |
| 12 | Shipping Mode | str | First/Second/Standard | **Transport Cost** |

---

## 5. Merged Schema (Output)

After ETL and merge, the unified dataset will have ~40 columns. Key ones:

```
timestamp               datetime
shipment_id             str
transport_mode          str (Sea|Air|Rail|Road)
product_category        str
origin_region           str
destination_region      str
distance_km             float
weight_mt               float

# Pillar 1: Delivery Delay
eta_variation_hours     float
days_for_shipping_real  int
days_for_shipping_sched int
late_delivery_risk      int
delivery_time_deviation float

# Pillar 2: Weather
weather_condition_severity  float
weather_conditions          str (Clear|Stormy|Rainy|Fog)
iot_temperature             float

# Pillar 3: Port Congestion
port_congestion_level       float
traffic_congestion_level    float
loading_unloading_time      float
waiting_time                float

# Pillar 4: Supplier Reliability
supplier_reliability_score  float
carrier_reliability         float
geopolitical_risk_score     float
driver_behavior_score       float
handling_equipment_avail    int

# Pillar 5: Transport Cost
shipping_costs              float
fuel_price_index            float
fuel_consumption_rate       float
benefit_per_order           float

# Pillar 6: Customs Delay
customs_clearance_time      float

# Pillar 7: Inventory Level
warehouse_inventory_level   float
historical_demand           float

# Engineered Targets
health_score                int (0–100)
risk_level                  str (Low|Medium|High|Critical)
```

---

## 6. Health Score Label Derivation

Each pillar maps to a penalty sub-score (0–100). The final score is:

```
health_score = max(0, 100 - Σ(wi * penalty_i))
```

### Default Weights (Domain Heuristic)

| Pillar | Weight | Rationale |
|---|---|---|
| Delivery Delay | 0.25 | Timeliness is #1 concern |
| Weather | 0.10 | Significant but seasonal |
| Port Congestion | 0.15 | Major modern bottleneck |
| Supplier Reliability | 0.20 | Upstream quality drives downstream |
| Transport Cost | 0.10 | Cost matters but secondary to reliability |
| Customs Delay | 0.10 | Critical for cross-border |
| Inventory Level | 0.10 | Buffer against disruptions |

Weights refined via regression on validation targets (`disruption_likelihood_score`, `delay_probability`).
