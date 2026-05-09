from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    HVAC_PRESETS,
    SCENARIOS,
    SEVERITY_LEVELS,
    CLIMATE_LEVELS,
    run_scenario_model,
    train_surrogate_models,
    run_early_sensitivity_analysis,
    run_robustness_analysis,
    simulate_combo,
    _load_base_weather,
    aggregate_zone_occupancy,
)
from report_addons import (
    read_weather_upload,
    build_detailed_tables,
    save_detailed_outputs,
    load_validation_file,
    build_validation_comparison,
    create_zip_from_folder,
    find_result_paths,
    setup_to_json_bytes,
    setup_from_upload,
    build_heat_exchanger_diagnostics,
    build_part_load_curve_analysis,
    build_latent_load_analysis,
    build_native_zone_load_table,
    build_formal_validation_metrics,
    build_global_sensitivity_from_samples,
    build_operation_schedule_template,
    validate_operation_schedule,
    run_multi_objective_search,
    build_advanced_control_candidates,
    build_control_objective_table,
    build_mpc_experimental_template,
    build_rl_experimental_dataset_spec,
)

st.set_page_config(page_title="HVAC ROM-Degradation Suite", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {background: linear-gradient(180deg, #07101f 0%, #101729 55%, #151827 100%);} 
.block-container {padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1360px;}
h1, h2, h3, h4, h5, h6, p, label, span, div {color: #eaf0fb;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);} 
div[data-baseweb="tab-list"] {gap: 0.55rem; border-bottom: 1px solid rgba(255,255,255,0.10); padding-bottom: 0.25rem;}
button[data-baseweb="tab"] {background: rgba(255,255,255,0.035) !important; border-radius: 14px 14px 0 0 !important; padding: 0.8rem 1.0rem !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.07) !important;}
button[data-baseweb="tab"][aria-selected="true"] {color: #ff686b !important; border-bottom: 2px solid #ff686b !important; background: rgba(255,255,255,0.075) !important;}
div[data-testid="stExpander"] {border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; background: rgba(255,255,255,0.035); margin-bottom: 0.9rem;}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 0.5rem 0.7rem;}
div[data-testid="stDataFrame"] {border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden;}
div.stButton > button {border-radius: 14px !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.18) !important;}
.small-muted {color:#aeb8ce; font-size:0.94rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div style="padding: 0.55rem 0 1.0rem 0;">
      <div style="font-size: 2.7rem; font-weight: 850; letter-spacing: -0.035em; color: #f6f8fc;">
        HVAC ROM-Degradation Suite
      </div>
      <div class="small-muted" style="max-width: 1040px; margin-top:0.35rem;">
        Reduced-order HVAC energy, degradation, EMS, scheduling, sensitivity, robustness, optimization, and fully coupled diagnostic-module modelling platform.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def default_zone_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"zone_name": "Lecture_01", "zone_type": "Lecture", "area_m2": 200.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
        {"zone_name": "Office_01", "zone_type": "Office", "area_m2": 120.0, "occ_density": 0.06, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
        {"zone_name": "Lab_01", "zone_type": "Lab", "area_m2": 180.0, "occ_density": 0.08, "term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
        {"zone_name": "Corridor", "zone_type": "Corridor", "area_m2": 100.0, "occ_density": 0.01, "term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
        {"zone_name": "Service_01", "zone_type": "Service", "area_m2": 80.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
    ])


def download_file_button(path: str | Path, label: str, key: str | None = None):
    path = Path(path)
    if path.exists() and path.is_file():
        with path.open("rb") as f:
            st.download_button(label, f.read(), file_name=path.name, key=key or f"dl_{path.name}")


def apply_setup_dict(data: dict):
    for k, v in data.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                st.session_state[kk] = vv
        else:
            st.session_state[k] = v


BUILT_IN_SETUPS = {
    "Educational medium building": {
        "building_type": "Educational / University building", "location": "User-defined", "area_m2": 5000.0, "floors": 4, "n_spaces": 40,
        "occupancy_density": 0.08, "lighting_w_m2": 10.0, "equipment_w_m2": 8.0, "sensible_w_per_person": 75.0,
        "airflow_m3h_m2": 4.0, "cooling_w_m2": 100.0, "heating_w_m2": 55.0,
        "wall_u": 0.60, "roof_u": 0.35, "window_u": 2.70, "shgc": 0.35, "glazing_ratio": 0.30, "infiltration_ach": 0.50,
        "hvac_system_type": "Chiller_AHU",
    },
    "Small office / training center": {
        "building_type": "Office / Training center", "location": "User-defined", "area_m2": 1200.0, "floors": 3, "n_spaces": 18,
        "occupancy_density": 0.06, "lighting_w_m2": 9.0, "equipment_w_m2": 11.0, "sensible_w_per_person": 75.0,
        "airflow_m3h_m2": 3.5, "cooling_w_m2": 95.0, "heating_w_m2": 45.0,
        "wall_u": 0.65, "roof_u": 0.40, "window_u": 2.80, "shgc": 0.38, "glazing_ratio": 0.25, "infiltration_ach": 0.45,
        "hvac_system_type": "VRF",
    },
}


def current_setup_dict() -> dict:
    keys = [
        "building_type", "location", "area_m2", "floors", "n_spaces", "occupancy_density", "lighting_w_m2", "equipment_w_m2", "sensible_w_per_person",
        "airflow_m3h_m2", "cooling_w_m2", "heating_w_m2", "wall_u", "roof_u", "window_u", "shgc", "glazing_ratio", "infiltration_ach",
        "hvac_system_type", "use_hvac_preset", "cop_cool_nom", "cop_heat_nom", "fan_eff", "pump_specific_w_m2", "auxiliary_w_m2", "dp_clean", "dp_warn", "dp_thresh", "dp_max",
        "years", "time_step_label", "t_set", "t_sp_min", "t_sp_max", "af_min", "af_max", "cop_aging_rate", "rf_star", "b_foul", "dust_rate", "k_clog", "deg_trigger",
        "filter_interval", "hx_interval", "e_price", "co2_factor", "cost_filter", "cost_hx", "degradation_model", "linear_deg_per_day", "exp_deg_rate_per_day",
        "APPLY_PART_LOAD_COP_TO_CORE", "APPLY_LATENT_LOAD_TO_CORE", "APPLY_HX_AIR_PRESSURE_TO_FAN", "APPLY_HX_WATER_PRESSURE_TO_PUMP", "APPLY_HX_UA_TO_CAPACITY", "APPLY_NATIVE_ZONE_LOADS",
        "PLR_CURVE_TYPE", "PLR_A", "PLR_B", "PLR_C", "PLR_D", "INDOOR_RH_TARGET_PCT", "LATENT_VENTILATION_FRACTION", "HX_WATER_DP_CLEAN_KPA", "HX_UA_LOSS_FACTOR",
    ]
    return {k: st.session_state.get(k) for k in keys if k in st.session_state}


def cfg_with_switches(cfg: HVACConfig, switches: dict[str, bool]) -> HVACConfig:
    for attr, val in switches.items():
        if hasattr(cfg, attr):
            setattr(cfg, attr, bool(val))
    return cfg


def build_weather_controls(prefix: str = "main"):
    c1, c2, c3 = st.columns([1.1, 1.0, 1.0])
    weather_mode_ui = c1.selectbox(
        "Weather source",
        ["synthetic", "upload_csv_epw", "epw_path", "csv_path"],
        format_func=lambda x: {"synthetic": "Synthetic weather at selected time step", "upload_csv_epw": "Upload CSV/EPW directly", "epw_path": "EPW path", "csv_path": "CSV path"}[x],
        key=f"{prefix}_weather_mode",
    )
    random_state = int(c2.number_input("Random state", min_value=1, value=42, step=1, key=f"{prefix}_random_state"))
    out_dir = c3.text_input("Output folder", f"{prefix}_run", key=f"{prefix}_out_dir")
    weather_df = None
    epw_path = None
    csv_path = None
    if weather_mode_ui == "upload_csv_epw":
        uploaded_weather = st.file_uploader("Upload weather file (.csv or .epw)", type=["csv", "epw", "txt"], key=f"{prefix}_weather_upload")
        if uploaded_weather is not None:
            try:
                weather_df = read_weather_upload(uploaded_weather)
                st.session_state[f"{prefix}_uploaded_weather_df"] = weather_df
                st.success(f"Weather upload parsed successfully: {len(weather_df)} records. Timestamped CSV/EPW files are preserved for native sub-daily simulation; daily files are expanded only when needed.")
                st.dataframe(weather_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Weather upload error: {e}")
        else:
            weather_df = st.session_state.get(f"{prefix}_uploaded_weather_df")
    elif weather_mode_ui == "epw_path":
        epw_path = st.text_input("EPW file path", "", key=f"{prefix}_epw_path")
    elif weather_mode_ui == "csv_path":
        csv_path = st.text_input("CSV weather file path", "", key=f"{prefix}_csv_path")
    engine_weather_mode = {"synthetic": "synthetic", "upload_csv_epw": "uploaded", "epw_path": "epw", "csv_path": "csv"}[weather_mode_ui]
    return engine_weather_mode, epw_path, csv_path, weather_df, random_state, out_dir


# Defaults
for k, v in BUILT_IN_SETUPS["Educational medium building"].items():
    st.session_state.setdefault(k, v)

# Main tabs: setup comes first by design.
tabs = st.tabs([
    "Building Identity & Setup",
    "Parameter Switches",
    "EMS Control Strategies",
    "Operation Scheduling",
    "Multi-Objective Optimization",
    "Scenario Modeling",
    "Sensitivity & Robustness",
    "Extra UI Tools",
    "KPI Charts",
    "Surrogate Train / Predict",
    "Exports",
    "Guide",
    "Model Validation",
    "Heat Exchanger Diagnostics",
    "Part-Load COP Curves",
    "Latent Cooling Load",
    "Zone-Level Load Analysis",
    "Global Sensitivity",
    "Advanced Plot Studio",
    "Advanced HVAC Control Library",
    "Core KPI Impact Dashboard",
    "Live Core Solver Lab",
])

with tabs[0]:
    st.subheader("Building identity and configuration setup")
    c1, c2, c3 = st.columns([1.1, 1.2, 1.0])
    preset_name = c1.selectbox("Saved / built-in setup", list(BUILT_IN_SETUPS.keys()), key="preset_selector")
    if c2.button("Apply selected setup"):
        apply_setup_dict(BUILT_IN_SETUPS[preset_name])
        st.success(f"Applied setup: {preset_name}")
    upload_setup = c3.file_uploader("Upload setup JSON", type=["json"], key="setup_upload")
    if upload_setup is not None:
        try:
            apply_setup_dict(setup_from_upload(upload_setup))
            st.success("Setup JSON loaded into the current session.")
        except Exception as e:
            st.error(str(e))
    st.download_button("Download current setup JSON", setup_to_json_bytes(current_setup_dict()), file_name="building_setup.json")

    st.markdown("### 1. Building identity")
    c1, c2 = st.columns(2)
    building_type = c1.text_input("Building type", key="building_type")
    location = c2.text_input("Location / weather label", key="location")

    st.markdown("### 2. Geometry")
    c1, c2, c3 = st.columns(3)
    area_m2 = c1.number_input("Conditioned area (m²)", min_value=100.0, step=100.0, key="area_m2")
    floors = c2.number_input("Floors", min_value=1, step=1, key="floors")
    n_spaces = c3.number_input("Number of spaces", min_value=1, step=1, key="n_spaces")

    st.markdown("### 3. Envelope")
    c1, c2, c3 = st.columns(3)
    wall_u = c1.number_input("Wall U-value (W/m²K)", min_value=0.01, step=0.05, key="wall_u")
    roof_u = c2.number_input("Roof U-value (W/m²K)", min_value=0.01, step=0.05, key="roof_u")
    window_u = c3.number_input("Window U-value (W/m²K)", min_value=0.01, step=0.1, key="window_u")
    c1, c2, c3 = st.columns(3)
    shgc = c1.number_input("SHGC", min_value=0.01, max_value=0.95, step=0.01, key="shgc")
    glazing_ratio = c2.number_input("Glazing ratio", min_value=0.01, max_value=0.95, step=0.01, key="glazing_ratio")
    infiltration_ach = c3.number_input("Infiltration (ACH)", min_value=0.0, step=0.1, key="infiltration_ach")

    st.markdown("### 4. Internal loads")
    c1, c2, c3, c4 = st.columns(4)
    occupancy_density = c1.number_input("Occupancy density (person/m²)", min_value=0.0001, step=0.01, format="%.4f", key="occupancy_density")
    lighting_w_m2 = c2.number_input("Lighting power density (W/m²)", min_value=0.0, step=1.0, key="lighting_w_m2")
    equipment_w_m2 = c3.number_input("Equipment power density (W/m²)", min_value=0.0, step=1.0, key="equipment_w_m2")
    sensible_w_per_person = c4.number_input("Sensible heat/person (W)", min_value=1.0, step=5.0, key="sensible_w_per_person")

    st.markdown("### 5. HVAC sizing and component")
    c1, c2, c3 = st.columns(3)
    hvac_system_type = c1.selectbox("HVAC system type", list(HVAC_PRESETS.keys()), key="hvac_system_type")
    use_hvac_preset = c2.checkbox("Apply selected HVAC preset", value=st.session_state.get("use_hvac_preset", True), key="use_hvac_preset")
    years = c3.number_input("Simulation years", min_value=1, max_value=50, step=1, key="years")

    c1, c2, c3 = st.columns(3)
    airflow_m3h_m2 = c1.number_input("Airflow intensity (m³/h·m²)", min_value=0.01, step=0.1, key="airflow_m3h_m2")
    cooling_w_m2 = c2.number_input("Cooling design intensity (W/m²)", min_value=1.0, step=5.0, key="cooling_w_m2")
    heating_w_m2 = c3.number_input("Heating design intensity (W/m²)", min_value=1.0, step=5.0, key="heating_w_m2")

    c1, c2, c3 = st.columns(3)
    cop_cool_nom = c1.number_input("Nominal cooling COP", min_value=0.8, value=float(st.session_state.get("cop_cool_nom", 4.5)), step=0.1, key="cop_cool_nom")
    cop_heat_nom = c2.number_input("Nominal heating COP", min_value=0.8, value=float(st.session_state.get("cop_heat_nom", 3.2)), step=0.1, key="cop_heat_nom")
    fan_eff = c3.number_input("Fan total efficiency", min_value=0.1, max_value=0.95, value=float(st.session_state.get("fan_eff", 0.70)), step=0.01, key="fan_eff")

    c1, c2 = st.columns(2)
    pump_specific_w_m2 = c1.number_input("Pump specific power (W/m²)", min_value=0.0, value=float(st.session_state.get("pump_specific_w_m2", 1.30)), step=0.05, key="pump_specific_w_m2")
    auxiliary_w_m2 = c2.number_input("Auxiliary power density (W/m²)", min_value=0.0, value=float(st.session_state.get("auxiliary_w_m2", 0.55)), step=0.05, key="auxiliary_w_m2")
    st.caption("Pump and auxiliary power are included in total energy when enabled in Parameter Switches. Presets can overwrite these values unless Custom/disabled preset is selected.")

    with st.expander("Strong coupled-module parameters for publication version", expanded=False):
        st.markdown("These parameters are used only when the corresponding coupled switch is enabled in the Parameter Switches tab.")
        c1, c2, c3, c4 = st.columns(4)
        st.session_state["PLR_CURVE_TYPE"] = c1.selectbox("Core PLR curve type", ["Linear", "Quadratic", "Cubic"], index=["Linear", "Quadratic", "Cubic"].index(st.session_state.get("PLR_CURVE_TYPE", "Quadratic")) if st.session_state.get("PLR_CURVE_TYPE", "Quadratic") in ["Linear", "Quadratic", "Cubic"] else 1)
        st.session_state["PLR_A"] = c2.number_input("PLR coefficient a", -5.0, 5.0, float(st.session_state.get("PLR_A", 0.85)), 0.05)
        st.session_state["PLR_B"] = c3.number_input("PLR coefficient b", -5.0, 5.0, float(st.session_state.get("PLR_B", 0.25)), 0.05)
        st.session_state["PLR_C"] = c4.number_input("PLR coefficient c", -5.0, 5.0, float(st.session_state.get("PLR_C", -0.10)), 0.05)
        st.session_state["PLR_D"] = st.number_input("PLR coefficient d", -5.0, 5.0, float(st.session_state.get("PLR_D", 0.0)), 0.05)
        c1, c2, c3 = st.columns(3)
        st.session_state["INDOOR_RH_TARGET_PCT"] = c1.number_input("Indoor RH target (%)", 10.0, 90.0, float(st.session_state.get("INDOOR_RH_TARGET_PCT", 50.0)), 1.0)
        st.session_state["LATENT_VENTILATION_FRACTION"] = c2.number_input("Outdoor-air fraction for latent load", 0.0, 1.0, float(st.session_state.get("LATENT_VENTILATION_FRACTION", 0.35)), 0.05)
        st.session_state["FLOOR_TO_FLOOR_M"] = c3.number_input("Floor-to-floor height (m)", 2.0, 6.0, float(st.session_state.get("FLOOR_TO_FLOOR_M", 3.2)), 0.1)
        c1, c2, c3, c4 = st.columns(4)
        st.session_state["HX_AIR_FOULING_FACTOR"] = c1.number_input("HX air fouling ΔP factor", 0.0, 5.0, float(st.session_state.get("HX_AIR_FOULING_FACTOR", 0.75)), 0.05)
        st.session_state["HX_WATER_DP_CLEAN_KPA"] = c2.number_input("Water-side clean ΔP (kPa)", 0.0, 500.0, float(st.session_state.get("HX_WATER_DP_CLEAN_KPA", 35.0)), 1.0)
        st.session_state["HX_WATER_FOULING_FACTOR"] = c3.number_input("Water-side fouling ΔP factor", 0.0, 5.0, float(st.session_state.get("HX_WATER_FOULING_FACTOR", 0.35)), 0.05)
        st.session_state["HX_PUMP_EFF"] = c4.number_input("Detailed pump efficiency", 0.05, 0.95, float(st.session_state.get("HX_PUMP_EFF", 0.65)), 0.01)
        c1, c2, c3, c4 = st.columns(4)
        st.session_state["HX_WATER_FLOW_M3H"] = c1.number_input("Water flow override (m³/h, 0=auto)", 0.0, 10000.0, float(st.session_state.get("HX_WATER_FLOW_M3H", 0.0)), 0.1)
        st.session_state["HX_WATER_FLOW_NOM_M3H"] = c2.number_input("Nominal water flow (m³/h, 0=auto)", 0.0, 10000.0, float(st.session_state.get("HX_WATER_FLOW_NOM_M3H", 0.0)), 0.1)
        st.session_state["HX_CHW_DT_K"] = c3.number_input("Chilled-water ΔT (K)", 1.0, 20.0, float(st.session_state.get("HX_CHW_DT_K", 5.0)), 0.5)
        st.session_state["HX_HW_DT_K"] = c4.number_input("Hot-water ΔT (K)", 1.0, 30.0, float(st.session_state.get("HX_HW_DT_K", 10.0)), 0.5)
        c1, c2 = st.columns(2)
        st.session_state["HX_UA_CLEAN_KW_K"] = c1.number_input("HX clean UA (kW/K, optional)", 0.0, 100000.0, float(st.session_state.get("HX_UA_CLEAN_KW_K", 0.0)), 1.0)
        st.session_state["HX_UA_LOSS_FACTOR"] = c2.number_input("UA loss factor at full degradation", 0.0, 0.95, float(st.session_state.get("HX_UA_LOSS_FACTOR", 0.30)), 0.05)

    c1, c2, c3, c4 = st.columns(4)
    dp_clean = c1.number_input("Clean static pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_clean", 150.0)), step=10.0, key="dp_clean")
    dp_warn = c2.number_input("Warning pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_warn", 320.0)), step=10.0, key="dp_warn")
    dp_thresh = c3.number_input("Replacement threshold pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_thresh", 420.0)), step=10.0, key="dp_thresh")
    dp_max = c4.number_input("Maximum pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_max", 450.0)), step=10.0, key="dp_max")

    st.markdown("### 6. Time-series, controls, cost, carbon")
    c1, c2, c3 = st.columns(3)
    time_step_label = c1.selectbox("Calculation time step", ["Daily", "12-hour", "6-hour", "3-hour", "Hourly"], key="time_step_label")
    time_step_hours = {"Daily": 24.0, "12-hour": 12.0, "6-hour": 6.0, "3-hour": 3.0, "Hourly": 1.0}[time_step_label]
    t_set = c2.number_input("Main setpoint T_SET (°C)", min_value=16.0, max_value=30.0, value=float(st.session_state.get("t_set", 23.0)), step=0.5, key="t_set")
    e_price = c3.number_input("Electricity price ($/kWh)", min_value=0.0, value=float(st.session_state.get("e_price", 0.12)), step=0.01, key="e_price")
    c1, c2, c3, c4 = st.columns(4)
    t_sp_min = c1.number_input("S3 min setpoint (°C)", min_value=16.0, max_value=30.0, value=float(st.session_state.get("t_sp_min", 21.0)), step=0.5, key="t_sp_min")
    t_sp_max = c2.number_input("S3 max setpoint (°C)", min_value=16.0, max_value=30.0, value=float(st.session_state.get("t_sp_max", 26.0)), step=0.5, key="t_sp_max")
    af_min = c3.number_input("S3 min airflow factor", min_value=0.1, max_value=1.0, value=float(st.session_state.get("af_min", 0.55)), step=0.05, key="af_min")
    af_max = c4.number_input("S3 max airflow factor", min_value=0.1, max_value=1.5, value=float(st.session_state.get("af_max", 1.0)), step=0.05, key="af_max")
    c1, c2, c3, c4 = st.columns(4)
    co2_factor = c1.number_input("CO₂ factor (kg/kWh)", min_value=0.0, value=float(st.session_state.get("co2_factor", 0.536)), step=0.01, key="co2_factor")
    cost_filter = c2.number_input("Filter cost", min_value=0.0, value=float(st.session_state.get("cost_filter", 50.0)), step=5.0, key="cost_filter")
    cost_hx = c3.number_input("HX cleaning cost", min_value=0.0, value=float(st.session_state.get("cost_hx", 300.0)), step=10.0, key="cost_hx")
    filter_interval = c4.number_input("Filter interval (days)", min_value=1, value=int(st.session_state.get("filter_interval", 90)), step=1, key="filter_interval")
    hx_interval = st.number_input("HX cleaning interval (days)", min_value=1, value=int(st.session_state.get("hx_interval", 180)), step=1, key="hx_interval")

    st.markdown("### 7. Degradation parameters")
    c1, c2, c3 = st.columns(3)
    degradation_model = c1.selectbox("Degradation model", ["physics", "linear_ts", "exponential_ts"], key="degradation_model", format_func=lambda x: {"physics":"Physics-based fouling/clogging", "linear_ts":"Linear time-series", "exponential_ts":"Exponential time-series"}[x])
    cop_aging_rate = c2.number_input("COP aging rate", min_value=0.0, value=float(st.session_state.get("cop_aging_rate", 0.005)), step=0.001, format="%.4f", key="cop_aging_rate")
    deg_trigger = c3.number_input("Degradation trigger", min_value=0.0, max_value=1.5, value=float(st.session_state.get("deg_trigger", 0.55)), step=0.01, key="deg_trigger")
    c1, c2, c3, c4 = st.columns(4)
    rf_star = c1.number_input("RF* fouling asymptote", min_value=1e-8, value=float(st.session_state.get("rf_star", 2e-4)), format="%.7f", key="rf_star")
    b_foul = c2.number_input("Fouling growth constant B", min_value=0.0, value=float(st.session_state.get("b_foul", 0.015)), step=0.001, format="%.4f", key="b_foul")
    dust_rate = c3.number_input("Dust accumulation rate", min_value=0.0, value=float(st.session_state.get("dust_rate", 1.2)), step=0.1, key="dust_rate")
    k_clog = c4.number_input("Clogging coefficient", min_value=0.0, value=float(st.session_state.get("k_clog", 6.0)), step=0.1, key="k_clog")
    c1, c2 = st.columns(2)
    linear_deg_per_day = c1.number_input("Linear degradation slope per day", min_value=0.0, value=float(st.session_state.get("linear_deg_per_day", 0.00012)), step=0.00001, format="%.6f", key="linear_deg_per_day")
    exp_deg_rate_per_day = c2.number_input("Exponential degradation rate per day", min_value=0.0, value=float(st.session_state.get("exp_deg_rate_per_day", 0.00018)), step=0.00001, format="%.6f", key="exp_deg_rate_per_day")

with tabs[1]:
    st.subheader("Parameter switches / quick control")
    st.markdown("These switches are passed into `HVACConfig` and affect the engine calculation directly. They do not create a duplicate model in the UI.")
    quick = st.columns(4)
    if quick[0].button("Enable all main terms"):
        for k in ["USE_ENVELOPE", "USE_WALLS", "USE_ROOF", "USE_WINDOWS", "USE_SOLAR", "USE_INFILTRATION", "USE_INTERNAL_GAINS", "USE_PEOPLE_GAINS", "USE_LIGHTING_GAINS", "USE_EQUIPMENT_GAINS", "USE_HVAC_FANS", "USE_HVAC_PUMPS", "USE_HVAC_AUXILIARY", "USE_COOLING", "USE_HEATING", "USE_DEGRADATION", "USE_CARBON", "USE_MAINTENANCE_COST", "APPLY_PART_LOAD_COP_TO_CORE", "APPLY_LATENT_LOAD_TO_CORE", "APPLY_HX_AIR_PRESSURE_TO_FAN", "APPLY_HX_WATER_PRESSURE_TO_PUMP", "APPLY_HX_UA_TO_CAPACITY", "APPLY_NATIVE_ZONE_LOADS"]:
            st.session_state[k] = True
    if quick[1].button("Thermal only: no degradation"):
        st.session_state["USE_DEGRADATION"] = False
        st.session_state["USE_MAINTENANCE_COST"] = False
    if quick[2].button("Envelope + weather only"):
        for k in ["USE_INTERNAL_GAINS", "USE_PEOPLE_GAINS", "USE_LIGHTING_GAINS", "USE_EQUIPMENT_GAINS", "USE_DEGRADATION", "USE_MAINTENANCE_COST"]:
            st.session_state[k] = False
    if quick[3].button("Disable optional post-tools"):
        for k in ["post_zone_analysis", "post_validation", "post_benchmark", "post_surrogate"]:
            st.session_state[k] = False
    if st.button("Reset coupled modules OFF / reproduce original core"):
        for k in ["APPLY_PART_LOAD_COP_TO_CORE", "APPLY_LATENT_LOAD_TO_CORE", "APPLY_HX_AIR_PRESSURE_TO_FAN", "APPLY_HX_WATER_PRESSURE_TO_PUMP", "APPLY_HX_UA_TO_CAPACITY", "APPLY_NATIVE_ZONE_LOADS"]:
            st.session_state[k] = False

    st.markdown("### Core physics switches")
    switch_names = [
        ("USE_ENVELOPE", "Envelope terms"), ("USE_WALLS", "Walls"), ("USE_ROOF", "Roof"), ("USE_WINDOWS", "Windows"),
        ("USE_SOLAR", "Solar gains"), ("USE_INFILTRATION", "Infiltration"), ("USE_INTERNAL_GAINS", "Internal gains"),
        ("USE_PEOPLE_GAINS", "People gains"), ("USE_LIGHTING_GAINS", "Lighting gains"), ("USE_EQUIPMENT_GAINS", "Equipment gains"),
        ("USE_HVAC_FANS", "HVAC fan energy"), ("USE_HVAC_PUMPS", "HVAC pump energy"), ("USE_HVAC_AUXILIARY", "HVAC auxiliary energy"),
        ("USE_COOLING", "Cooling"), ("USE_HEATING", "Heating"),
        ("USE_DEGRADATION", "Degradation"), ("USE_CARBON", "Carbon"), ("USE_MAINTENANCE_COST", "Maintenance cost"),
        ("APPLY_PART_LOAD_COP_TO_CORE", "Apply part-load COP to core"),
        ("APPLY_LATENT_LOAD_TO_CORE", "Apply latent cooling to core"),
        ("APPLY_HX_AIR_PRESSURE_TO_FAN", "Apply HX air ΔP to fan"),
        ("APPLY_HX_WATER_PRESSURE_TO_PUMP", "Apply HX water ΔP to pump"),
        ("APPLY_HX_UA_TO_CAPACITY", "Apply HX UA degradation to capacity"),
        ("APPLY_NATIVE_ZONE_LOADS", "Use native zone-by-zone load sum"),
    ]
    cols = st.columns(4)
    switches = {}
    for i, (key, label) in enumerate(switch_names):
        st.session_state.setdefault(key, True)
        with cols[i % 4]:
            switches[key] = st.checkbox(label, key=key)
    st.info("When the coupled switches are enabled, the diagnostic modules modify the official Scenario Modeling results. Keep them OFF to reproduce the older publication-plus engine.")
    st.markdown("### Post-processing switches")
    c1, c2, c3, c4 = st.columns(4)
    post_zone = c1.checkbox("Zone analysis", value=st.session_state.get("post_zone_analysis", True), key="post_zone_analysis")
    post_validation = c2.checkbox("Validation", value=st.session_state.get("post_validation", True), key="post_validation")
    post_benchmark = c3.checkbox("Benchmark/sensitivity sheets", value=st.session_state.get("post_benchmark", True), key="post_benchmark")
    post_surrogate = c4.checkbox("Surrogate modelling", value=st.session_state.get("post_surrogate", True), key="post_surrogate")

# Build engine objects after setup/switches are defined.
bldg = BuildingSpec(
    building_type=st.session_state.get("building_type", "Educational / University building"),
    location=st.session_state.get("location", "User-defined"),
    conditioned_area_m2=float(st.session_state.get("area_m2", 5000.0)),
    floors=int(st.session_state.get("floors", 4)),
    n_spaces=int(st.session_state.get("n_spaces", 40)),
    occupancy_density_p_m2=float(st.session_state.get("occupancy_density", 0.08)),
    lighting_w_m2=float(st.session_state.get("lighting_w_m2", 10.0)),
    equipment_w_m2=float(st.session_state.get("equipment_w_m2", 8.0)),
    airflow_m3h_m2=float(st.session_state.get("airflow_m3h_m2", 4.0)),
    infiltration_ach=float(st.session_state.get("infiltration_ach", 0.5)),
    sensible_w_per_person=float(st.session_state.get("sensible_w_per_person", 75.0)),
    cooling_intensity_w_m2=float(st.session_state.get("cooling_w_m2", 100.0)),
    heating_intensity_w_m2=float(st.session_state.get("heating_w_m2", 55.0)),
    wall_u=float(st.session_state.get("wall_u", 0.6)),
    roof_u=float(st.session_state.get("roof_u", 0.35)),
    window_u=float(st.session_state.get("window_u", 2.7)),
    shgc=float(st.session_state.get("shgc", 0.35)),
    glazing_ratio=float(st.session_state.get("glazing_ratio", 0.30)),
)
time_step_hours = {"Daily": 24.0, "12-hour": 12.0, "6-hour": 6.0, "3-hour": 3.0, "Hourly": 1.0}[st.session_state.get("time_step_label", "Daily")]
cfg = HVACConfig(
    years=int(st.session_state.get("years", 20)),
    hvac_system_type=st.session_state.get("hvac_system_type", "Chiller_AHU"),
    COP_COOL_NOM=float(st.session_state.get("cop_cool_nom", 4.5)),
    COP_HEAT_NOM=float(st.session_state.get("cop_heat_nom", 3.2)),
    FAN_EFF=float(st.session_state.get("fan_eff", 0.70)),
    PUMP_SPECIFIC_W_M2=float(st.session_state.get("pump_specific_w_m2", 1.30)),
    AUXILIARY_W_M2=float(st.session_state.get("auxiliary_w_m2", 0.55)),
    T_SET=float(st.session_state.get("t_set", 23.0)),
    T_SP_MIN=float(st.session_state.get("t_sp_min", 21.0)),
    T_SP_MAX=float(st.session_state.get("t_sp_max", 26.0)),
    AF_MIN=float(st.session_state.get("af_min", 0.55)),
    AF_MAX=float(st.session_state.get("af_max", 1.0)),
    DP_CLEAN=float(st.session_state.get("dp_clean", 150.0)),
    DP_WARN=float(st.session_state.get("dp_warn", 320.0)),
    DP_THRESH=float(st.session_state.get("dp_thresh", 420.0)),
    DP_MAX=float(st.session_state.get("dp_max", 450.0)),
    COP_AGING_RATE=float(st.session_state.get("cop_aging_rate", 0.005)),
    RF_STAR=float(st.session_state.get("rf_star", 2e-4)),
    B_FOUL=float(st.session_state.get("b_foul", 0.015)),
    DUST_RATE=float(st.session_state.get("dust_rate", 1.2)),
    K_CLOG=float(st.session_state.get("k_clog", 6.0)),
    DEG_TRIGGER=float(st.session_state.get("deg_trigger", 0.55)),
    E_PRICE=float(st.session_state.get("e_price", 0.12)),
    CO2_FACTOR=float(st.session_state.get("co2_factor", 0.536)),
    COST_FILTER=float(st.session_state.get("cost_filter", 50.0)),
    COST_HX=float(st.session_state.get("cost_hx", 300.0)),
    FILTER_INTERVAL=int(st.session_state.get("filter_interval", 90)),
    HX_INTERVAL=int(st.session_state.get("hx_interval", 180)),
    degradation_model=st.session_state.get("degradation_model", "physics"),
    LINEAR_DEG_PER_DAY=float(st.session_state.get("linear_deg_per_day", 0.00012)),
    EXP_DEG_RATE_PER_DAY=float(st.session_state.get("exp_deg_rate_per_day", 0.00018)),
    TIME_STEP_HOURS=time_step_hours,
    USE_HVAC_PRESET=bool(st.session_state.get("use_hvac_preset", True)),
    APPLY_PART_LOAD_COP_TO_CORE=bool(st.session_state.get("APPLY_PART_LOAD_COP_TO_CORE", False)),
    APPLY_LATENT_LOAD_TO_CORE=bool(st.session_state.get("APPLY_LATENT_LOAD_TO_CORE", False)),
    APPLY_HX_AIR_PRESSURE_TO_FAN=bool(st.session_state.get("APPLY_HX_AIR_PRESSURE_TO_FAN", False)),
    APPLY_HX_WATER_PRESSURE_TO_PUMP=bool(st.session_state.get("APPLY_HX_WATER_PRESSURE_TO_PUMP", False)),
    APPLY_HX_UA_TO_CAPACITY=bool(st.session_state.get("APPLY_HX_UA_TO_CAPACITY", False)),
    APPLY_NATIVE_ZONE_LOADS=bool(st.session_state.get("APPLY_NATIVE_ZONE_LOADS", False)),
    PLR_CURVE_TYPE=st.session_state.get("PLR_CURVE_TYPE", "Quadratic"),
    PLR_A=float(st.session_state.get("PLR_A", 0.85)),
    PLR_B=float(st.session_state.get("PLR_B", 0.25)),
    PLR_C=float(st.session_state.get("PLR_C", -0.10)),
    PLR_D=float(st.session_state.get("PLR_D", 0.0)),
    INDOOR_RH_TARGET_PCT=float(st.session_state.get("INDOOR_RH_TARGET_PCT", 50.0)),
    LATENT_VENTILATION_FRACTION=float(st.session_state.get("LATENT_VENTILATION_FRACTION", 0.35)),
    FLOOR_TO_FLOOR_M=float(st.session_state.get("FLOOR_TO_FLOOR_M", 3.2)),
    HX_AIR_FOULING_FACTOR=float(st.session_state.get("HX_AIR_FOULING_FACTOR", 0.75)),
    HX_WATER_DP_CLEAN_KPA=float(st.session_state.get("HX_WATER_DP_CLEAN_KPA", 35.0)),
    HX_WATER_FLOW_M3H=float(st.session_state.get("HX_WATER_FLOW_M3H", 0.0)),
    HX_WATER_FLOW_NOM_M3H=float(st.session_state.get("HX_WATER_FLOW_NOM_M3H", 0.0)),
    HX_WATER_FOULING_FACTOR=float(st.session_state.get("HX_WATER_FOULING_FACTOR", 0.35)),
    HX_PUMP_EFF=float(st.session_state.get("HX_PUMP_EFF", 0.65)),
    HX_CHW_DT_K=float(st.session_state.get("HX_CHW_DT_K", 5.0)),
    HX_HW_DT_K=float(st.session_state.get("HX_HW_DT_K", 10.0)),
    HX_UA_CLEAN_KW_K=float(st.session_state.get("HX_UA_CLEAN_KW_K", 0.0)),
    HX_UA_LOSS_FACTOR=float(st.session_state.get("HX_UA_LOSS_FACTOR", 0.30)),
    EMS_MODE=st.session_state.get("EMS_MODE", "Disabled"),
    EMS_OCC_CONTROL=bool(st.session_state.get("EMS_OCC_CONTROL", False)),
    EMS_NIGHT_SETBACK=bool(st.session_state.get("EMS_NIGHT_SETBACK", False)),
    EMS_DEMAND_RESPONSE=bool(st.session_state.get("EMS_DEMAND_RESPONSE", False)),
    EMS_ECONOMIZER=bool(st.session_state.get("EMS_ECONOMIZER", False)),
    EMS_OPTIMUM_START=bool(st.session_state.get("EMS_OPTIMUM_START", False)),
    EMS_CUSTOM_SCHEDULE_ENABLED=bool(st.session_state.get("EMS_CUSTOM_SCHEDULE_ENABLED", False)),
    EMS_LOW_OCC_THRESHOLD=float(st.session_state.get("EMS_LOW_OCC_THRESHOLD", 0.25)),
    EMS_LOW_OCC_AIRFLOW_FACTOR=float(st.session_state.get("EMS_LOW_OCC_AIRFLOW_FACTOR", 0.65)),
    EMS_LOW_OCC_SETPOINT_SHIFT_C=float(st.session_state.get("EMS_LOW_OCC_SETPOINT_SHIFT_C", 1.0)),
    EMS_NIGHT_START_HOUR=float(st.session_state.get("EMS_NIGHT_START_HOUR", 19.0)),
    EMS_NIGHT_END_HOUR=float(st.session_state.get("EMS_NIGHT_END_HOUR", 6.0)),
    EMS_NIGHT_SETPOINT_SHIFT_C=float(st.session_state.get("EMS_NIGHT_SETPOINT_SHIFT_C", 2.0)),
    EMS_NIGHT_AIRFLOW_FACTOR=float(st.session_state.get("EMS_NIGHT_AIRFLOW_FACTOR", 0.55)),
    EMS_DR_START_HOUR=float(st.session_state.get("EMS_DR_START_HOUR", 13.0)),
    EMS_DR_END_HOUR=float(st.session_state.get("EMS_DR_END_HOUR", 17.0)),
    EMS_DR_SETPOINT_SHIFT_C=float(st.session_state.get("EMS_DR_SETPOINT_SHIFT_C", 1.5)),
    EMS_DR_AIRFLOW_REDUCTION=float(st.session_state.get("EMS_DR_AIRFLOW_REDUCTION", 0.15)),
    EMS_ECONOMIZER_TEMP_LOW_C=float(st.session_state.get("EMS_ECONOMIZER_TEMP_LOW_C", 16.0)),
    EMS_ECONOMIZER_TEMP_HIGH_C=float(st.session_state.get("EMS_ECONOMIZER_TEMP_HIGH_C", 22.0)),
    EMS_ECONOMIZER_COOLING_REDUCTION=float(st.session_state.get("EMS_ECONOMIZER_COOLING_REDUCTION", 0.20)),
    EMS_OPTIMUM_START_HOUR=float(st.session_state.get("EMS_OPTIMUM_START_HOUR", 7.0)),
    EMS_PRECOOL_SHIFT_C=float(st.session_state.get("EMS_PRECOOL_SHIFT_C", -0.8)),
)
cfg = cfg_with_switches(cfg, {k: st.session_state.get(k, True) for k, _ in switch_names})


def _clone_engine_cfg(base_cfg: HVACConfig, years_override: int | None = None) -> HVACConfig:
    new_cfg = HVACConfig(**asdict(base_cfg))
    # Do not carry private/non-dataclass attributes that Streamlit may add later.
    if years_override is not None:
        new_cfg.years = int(years_override)
    return new_cfg


def _set_all_physics_switches(new_cfg: HVACConfig, value: bool = True) -> HVACConfig:
    for attr in [
        "USE_ENVELOPE", "USE_WALLS", "USE_ROOF", "USE_WINDOWS", "USE_SOLAR", "USE_INFILTRATION",
        "USE_INTERNAL_GAINS", "USE_PEOPLE_GAINS", "USE_LIGHTING_GAINS", "USE_EQUIPMENT_GAINS",
        "USE_HVAC_FANS", "USE_HVAC_PUMPS", "USE_HVAC_AUXILIARY", "USE_COOLING", "USE_HEATING",
        "USE_DEGRADATION", "USE_CARBON", "USE_MAINTENANCE_COST",
    ]:
        if hasattr(new_cfg, attr):
            setattr(new_cfg, attr, bool(value))
    return new_cfg


def _disable_coupled_modules(new_cfg: HVACConfig) -> HVACConfig:
    for attr in [
        "APPLY_PART_LOAD_COP_TO_CORE", "APPLY_LATENT_LOAD_TO_CORE", "APPLY_HX_AIR_PRESSURE_TO_FAN",
        "APPLY_HX_WATER_PRESSURE_TO_PUMP", "APPLY_HX_UA_TO_CAPACITY", "APPLY_NATIVE_ZONE_LOADS",
    ]:
        if hasattr(new_cfg, attr):
            setattr(new_cfg, attr, False)
    return new_cfg


def _disable_ems(new_cfg: HVACConfig) -> HVACConfig:
    for attr in [
        "EMS_OCC_CONTROL", "EMS_NIGHT_SETBACK", "EMS_DEMAND_RESPONSE", "EMS_ECONOMIZER",
        "EMS_OPTIMUM_START", "EMS_CUSTOM_SCHEDULE_ENABLED",
    ]:
        if hasattr(new_cfg, attr):
            setattr(new_cfg, attr, False)
    if hasattr(new_cfg, "EMS_MODE"):
        new_cfg.EMS_MODE = "Disabled"
    return new_cfg


def _apply_control_row_to_cfg(new_cfg: HVACConfig, row: dict) -> HVACConfig:
    new_cfg.EMS_MODE = str(row.get("ems_mode", "Smart hybrid"))
    new_cfg.EMS_OCC_CONTROL = bool(row.get("use_occ_reset", False))
    new_cfg.EMS_NIGHT_SETBACK = bool(row.get("use_night_setback", False))
    new_cfg.EMS_DEMAND_RESPONSE = bool(row.get("use_demand_response", False))
    new_cfg.EMS_ECONOMIZER = bool(row.get("use_economizer", False))
    new_cfg.EMS_OPTIMUM_START = bool(row.get("use_optimum_start", False))
    new_cfg.EMS_LOW_OCC_SETPOINT_SHIFT_C = float(row.get("setpoint_shift_C", getattr(new_cfg, "EMS_LOW_OCC_SETPOINT_SHIFT_C", 1.0)))
    new_cfg.EMS_LOW_OCC_AIRFLOW_FACTOR = float(row.get("airflow_factor", getattr(new_cfg, "EMS_LOW_OCC_AIRFLOW_FACTOR", 0.65)))
    new_cfg.EMS_ECONOMIZER_COOLING_REDUCTION = float(row.get("economizer_reduction", getattr(new_cfg, "EMS_ECONOMIZER_COOLING_REDUCTION", 0.2)))
    return new_cfg


def _impact_summary_from_csv(summary_csv: str | Path) -> dict:
    df = pd.read_csv(summary_csv)
    def _sum(col):
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if col in df.columns else 0.0
    def _mean(col):
        return float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df.columns else 0.0
    return {
        "Total Energy MWh": _sum("Total Energy MWh"),
        "Thermal HVAC MWh": _sum("Total Thermal HVAC Energy MWh"),
        "Fan MWh": _sum("Total Fan Energy MWh"),
        "Pump MWh": _sum("Total Pump Energy MWh"),
        "Auxiliary MWh": _sum("Total Auxiliary Energy MWh"),
        "Total CO2 tonne": _sum("Total CO2 tonne"),
        "Total Cost USD": _sum("Total Cost USD"),
        "Mean Comfort Deviation C": _mean("Mean Comfort Deviation C"),
        "Mean Degradation Index": _mean("Mean Degradation Index"),
        "Mean COP": _mean("Mean COP"),
    }


def _run_core_impact_pair(
    label: str,
    base_cfg: HVACConfig,
    modified_cfg: HVACConfig,
    impact_dir: str | Path,
    impact_years: int,
    impact_strategy: str,
    impact_severity: str,
    impact_climate: str,
    weather_mode: str = "synthetic",
    epw_path: str | None = None,
    csv_path: str | None = None,
    weather_df: pd.DataFrame | None = None,
    random_state: int = 42,
    operation_schedule_for_modified: pd.DataFrame | None = None,
) -> pd.DataFrame:
    impact_dir = Path(impact_dir)
    base_out = impact_dir / f"{label}_baseline"
    mod_out = impact_dir / f"{label}_modified"
    base_result = run_scenario_model(
        output_dir=base_out,
        axis_mode="one_severity",
        bldg=bldg,
        cfg=_clone_engine_cfg(base_cfg, impact_years),
        weather_mode=weather_mode,
        epw_path=epw_path,
        csv_path=csv_path,
        weather_df=weather_df,
        fixed_strategy=impact_strategy,
        fixed_severity=impact_severity,
        fixed_climate=impact_climate,
        zone_df=st.session_state.get("last_zone_df"),
        random_state=random_state,
        include_baseline_layer=False,
        degradation_model=getattr(base_cfg, "degradation_model", "physics"),
        time_step_hours=getattr(base_cfg, "TIME_STEP_HOURS", 24.0),
        operation_schedule_df=None,
    )
    mod_result = run_scenario_model(
        output_dir=mod_out,
        axis_mode="one_severity",
        bldg=bldg,
        cfg=_clone_engine_cfg(modified_cfg, impact_years),
        weather_mode=weather_mode,
        epw_path=epw_path,
        csv_path=csv_path,
        weather_df=weather_df,
        fixed_strategy=impact_strategy,
        fixed_severity=impact_severity,
        fixed_climate=impact_climate,
        zone_df=st.session_state.get("last_zone_df"),
        random_state=random_state,
        include_baseline_layer=False,
        degradation_model=getattr(modified_cfg, "degradation_model", "physics"),
        time_step_hours=getattr(modified_cfg, "TIME_STEP_HOURS", 24.0),
        operation_schedule_df=operation_schedule_for_modified,
    )
    base_k = _impact_summary_from_csv(base_result["summary_csv"])
    mod_k = _impact_summary_from_csv(mod_result["summary_csv"])
    rows = []
    for k in base_k:
        base_val = base_k[k]
        mod_val = mod_k[k]
        delta = mod_val - base_val
        pct = (delta / base_val * 100.0) if abs(base_val) > 1e-12 else 0.0
        rows.append({
            "impact_case": label,
            "kpi": k,
            "baseline": base_val,
            "modified": mod_val,
            "delta": delta,
            "delta_pct": pct,
        })
    out = pd.DataFrame(rows)
    impact_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(impact_dir / f"{label}_kpi_impact.csv", index=False)
    return out


with tabs[2]:
    st.subheader("EMS Control Strategies")
    st.markdown("Configure Energy Management System (EMS) overlays that modify setpoint, airflow, and selected cooling behavior during simulation. Defaults keep the original model unchanged.")
    c1, c2, c3 = st.columns(3)
    st.session_state["EMS_MODE"] = c1.selectbox(
        "EMS strategy mode",
        ["Disabled", "Occupancy-based", "Night setback", "Demand response", "Economizer", "Optimum start", "Smart hybrid", "Custom scheduled"],
        index=["Disabled", "Occupancy-based", "Night setback", "Demand response", "Economizer", "Optimum start", "Smart hybrid", "Custom scheduled"].index(st.session_state.get("EMS_MODE", "Disabled")) if st.session_state.get("EMS_MODE", "Disabled") in ["Disabled", "Occupancy-based", "Night setback", "Demand response", "Economizer", "Optimum start", "Smart hybrid", "Custom scheduled"] else 0,
    )
    st.session_state["EMS_OCC_CONTROL"] = c2.checkbox("Enable occupancy-based reset", value=bool(st.session_state.get("EMS_OCC_CONTROL", False)))
    st.session_state["EMS_NIGHT_SETBACK"] = c3.checkbox("Enable night setback", value=bool(st.session_state.get("EMS_NIGHT_SETBACK", False)))
    c1, c2, c3 = st.columns(3)
    st.session_state["EMS_DEMAND_RESPONSE"] = c1.checkbox("Enable demand-response event", value=bool(st.session_state.get("EMS_DEMAND_RESPONSE", False)))
    st.session_state["EMS_ECONOMIZER"] = c2.checkbox("Enable economizer/free-cooling logic", value=bool(st.session_state.get("EMS_ECONOMIZER", False)))
    st.session_state["EMS_OPTIMUM_START"] = c3.checkbox("Enable optimum start/pre-cooling", value=bool(st.session_state.get("EMS_OPTIMUM_START", False)))
    st.session_state["EMS_CUSTOM_SCHEDULE_ENABLED"] = st.checkbox("Use Operation Scheduling tab as custom EMS schedule", value=bool(st.session_state.get("EMS_CUSTOM_SCHEDULE_ENABLED", False)))

    st.markdown("### Occupancy-based control")
    c1, c2, c3 = st.columns(3)
    st.session_state["EMS_LOW_OCC_THRESHOLD"] = c1.number_input("Low occupancy threshold", 0.0, 1.0, float(st.session_state.get("EMS_LOW_OCC_THRESHOLD", 0.25)), 0.05)
    st.session_state["EMS_LOW_OCC_AIRFLOW_FACTOR"] = c2.number_input("Low occupancy airflow factor", 0.1, 1.5, float(st.session_state.get("EMS_LOW_OCC_AIRFLOW_FACTOR", 0.65)), 0.05)
    st.session_state["EMS_LOW_OCC_SETPOINT_SHIFT_C"] = c3.number_input("Low occupancy setpoint shift (°C)", -5.0, 5.0, float(st.session_state.get("EMS_LOW_OCC_SETPOINT_SHIFT_C", 1.0)), 0.25)

    st.markdown("### Night setback")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["EMS_NIGHT_START_HOUR"] = c1.number_input("Night start hour", 0.0, 24.0, float(st.session_state.get("EMS_NIGHT_START_HOUR", 19.0)), 0.5)
    st.session_state["EMS_NIGHT_END_HOUR"] = c2.number_input("Night end hour", 0.0, 24.0, float(st.session_state.get("EMS_NIGHT_END_HOUR", 6.0)), 0.5)
    st.session_state["EMS_NIGHT_SETPOINT_SHIFT_C"] = c3.number_input("Night setpoint shift (°C)", -5.0, 8.0, float(st.session_state.get("EMS_NIGHT_SETPOINT_SHIFT_C", 2.0)), 0.25)
    st.session_state["EMS_NIGHT_AIRFLOW_FACTOR"] = c4.number_input("Night airflow factor", 0.1, 1.5, float(st.session_state.get("EMS_NIGHT_AIRFLOW_FACTOR", 0.55)), 0.05)

    st.markdown("### Demand response")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["EMS_DR_START_HOUR"] = c1.number_input("DR start hour", 0.0, 24.0, float(st.session_state.get("EMS_DR_START_HOUR", 13.0)), 0.5)
    st.session_state["EMS_DR_END_HOUR"] = c2.number_input("DR end hour", 0.0, 24.0, float(st.session_state.get("EMS_DR_END_HOUR", 17.0)), 0.5)
    st.session_state["EMS_DR_SETPOINT_SHIFT_C"] = c3.number_input("DR setpoint shift (°C)", -2.0, 6.0, float(st.session_state.get("EMS_DR_SETPOINT_SHIFT_C", 1.5)), 0.25)
    st.session_state["EMS_DR_AIRFLOW_REDUCTION"] = c4.number_input("DR airflow reduction fraction", 0.0, 0.8, float(st.session_state.get("EMS_DR_AIRFLOW_REDUCTION", 0.15)), 0.05)

    st.markdown("### Economizer and optimum start")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["EMS_ECONOMIZER_TEMP_LOW_C"] = c1.number_input("Economizer low temp (°C)", -10.0, 35.0, float(st.session_state.get("EMS_ECONOMIZER_TEMP_LOW_C", 16.0)), 0.5)
    st.session_state["EMS_ECONOMIZER_TEMP_HIGH_C"] = c2.number_input("Economizer high temp (°C)", -10.0, 40.0, float(st.session_state.get("EMS_ECONOMIZER_TEMP_HIGH_C", 22.0)), 0.5)
    st.session_state["EMS_ECONOMIZER_COOLING_REDUCTION"] = c3.number_input("Economizer cooling reduction", 0.0, 0.8, float(st.session_state.get("EMS_ECONOMIZER_COOLING_REDUCTION", 0.20)), 0.05)
    st.session_state["EMS_OPTIMUM_START_HOUR"] = c4.number_input("Optimum start hour", 0.0, 24.0, float(st.session_state.get("EMS_OPTIMUM_START_HOUR", 7.0)), 0.5)
    st.session_state["EMS_PRECOOL_SHIFT_C"] = st.number_input("Optimum-start setpoint shift (°C)", -5.0, 5.0, float(st.session_state.get("EMS_PRECOOL_SHIFT_C", -0.8)), 0.25)
    st.info("EMS actions are written to the daily/time-step dataset as ems_active, ems_occ_control, ems_night_setback, ems_demand_response, ems_economizer, ems_custom_schedule, and ems_optimum_start.")

with tabs[3]:
    st.subheader("Schedule of Operations")
    st.markdown("Build an editable hourly operation schedule. When 'Use Operation Scheduling tab as custom EMS schedule' is enabled, the schedule modifies setpoint and airflow during the engine run.")
    if "operation_schedule_df" not in st.session_state:
        st.session_state["operation_schedule_df"] = build_operation_schedule_template()
    c1, c2 = st.columns(2)
    if c1.button("Load default educational schedule"):
        st.session_state["operation_schedule_df"] = build_operation_schedule_template()
    schedule_upload = c2.file_uploader("Upload operation schedule CSV", type=["csv"], key="schedule_upload")
    if schedule_upload is not None:
        st.session_state["operation_schedule_df"] = validate_operation_schedule(pd.read_csv(schedule_upload))
    sched = st.data_editor(st.session_state["operation_schedule_df"], num_rows="dynamic", use_container_width=True, key="operation_schedule_editor")
    sched = validate_operation_schedule(sched)
    st.session_state["operation_schedule_df"] = sched
    st.dataframe(sched, use_container_width=True)
    st.download_button("Download operation_schedule.csv", sched.to_csv(index=False).encode("utf-8"), file_name="operation_schedule.csv", mime="text/csv")
    if PLOTLY_AVAILABLE and not sched.empty:
        fig = px.timeline(
            sched.assign(Task=sched["day_type"] + " | AF=" + sched["airflow_factor"].round(2).astype(str)),
            x_start="start_hour", x_end="end_hour", y="Task", color="day_type", title="Operation schedule windows"
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Multi-Objective Optimization")
    st.markdown("Screen control/EMS candidates against energy, degradation, comfort, and carbon objectives. The tab supports built-in lightweight optimizers and records the optimizer label for future comparison with external algorithms.")
    c1, c2, c3 = st.columns(3)
    # Streamlit does not allow changing the value of a widget key after the widget
    # has been instantiated in the same script run. Control Library buttons therefore
    # write a pending value, which is applied here before the optimizer selectbox exists.
    if "pending_moo_optimizer" in st.session_state:
        st.session_state["moo_optimizer"] = st.session_state.pop("pending_moo_optimizer")
    optimizer_name = c1.selectbox("Optimizer", ["Weighted random search", "Grid search", "NSGA-II style screening", "Particle Swarm placeholder", "Genetic Algorithm placeholder", "Custom optimizer label"], key="moo_optimizer")
    if optimizer_name == "Custom optimizer label":
        optimizer_name = c1.text_input("Custom optimizer name", "My optimizer")
    n_candidates = int(c2.number_input("Candidates", min_value=2, max_value=60, value=10, step=1, key="moo_candidates"))
    moo_years = int(c3.number_input("Analysis years", min_value=1, max_value=10, value=1, step=1, key="moo_years"))
    c1, c2, c3 = st.columns(3)
    moo_strategy = c1.selectbox("Strategy", list(SCENARIOS.keys()), index=3, key="moo_strategy")
    moo_severity = c2.selectbox("Severity", list(SEVERITY_LEVELS.keys()), index=1, key="moo_severity")
    moo_climate = c3.selectbox("Climate", list(CLIMATE_LEVELS.keys()), key="moo_climate")
    st.markdown("### Objective weights")
    c1, c2, c3, c4 = st.columns(4)
    w_energy = c1.number_input("Energy weight", 0.0, 1.0, 0.35, 0.05, key="moo_w_energy")
    w_deg = c2.number_input("Degradation weight", 0.0, 1.0, 0.25, 0.05, key="moo_w_deg")
    w_comfort = c3.number_input("Comfort weight", 0.0, 1.0, 0.25, 0.05, key="moo_w_comfort")
    w_carbon = c4.number_input("Carbon weight", 0.0, 1.0, 0.15, 0.05, key="moo_w_carbon")
    engine_weather_mode_moo, epw_path_moo, csv_path_moo, weather_df_moo, random_state_moo, out_dir_moo = build_weather_controls("moo")
    out_dir_moo = str(Path(out_dir_moo) / "multi_objective")
    use_sched_moo = st.checkbox("Use current operation schedule in optimization", value=bool(st.session_state.get("EMS_CUSTOM_SCHEDULE_ENABLED", False)), key="moo_use_schedule")
    if st.button("Run multi-objective optimization", type="primary"):
        try:
            if engine_weather_mode_moo == "uploaded" and weather_df_moo is None:
                st.warning("Upload weather first, or select synthetic/path weather.")
                st.stop()
            result = run_multi_objective_search(
                output_dir=out_dir_moo,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode_moo,
                epw_path=epw_path_moo,
                csv_path=csv_path_moo,
                weather_df=weather_df_moo,
                operation_schedule_df=st.session_state.get("operation_schedule_df") if use_sched_moo else None,
                fixed_strategy=moo_strategy,
                fixed_severity=moo_severity,
                fixed_climate=moo_climate,
                optimizer_name=optimizer_name,
                n_candidates=n_candidates,
                analysis_years=moo_years,
                random_state=random_state_moo,
                weight_energy=w_energy,
                weight_degradation=w_deg,
                weight_comfort=w_comfort,
                weight_carbon=w_carbon,
            )
            st.success("Multi-objective search completed.")
            st.json(result)
            df = pd.read_csv(result["candidates_csv"])
            st.dataframe(df, use_container_width=True)
            download_file_button(result["candidates_csv"], "Download multi_objective_candidates.csv")
            download_file_button(result["pareto_csv"], "Download multi_objective_pareto.csv")
            if PLOTLY_AVAILABLE and not df.empty:
                fig = px.scatter(df, x="Total Energy MWh", y="Mean Comfort Deviation C", color="pareto_candidate", size="Total CO2 tonne", hover_data=["candidate_id", "weighted_objective"], title="Multi-objective candidate space")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[5]:
    st.subheader("Scenario modeling")
    c1, c2, c3 = st.columns(3)
    axis_mode = c1.selectbox(
        "Analysis mode",
        ["baseline_scenario", "one_severity", "one_strategy", "two_axis", "three_axis"],
        format_func=lambda x: {"baseline_scenario": "Baseline Scenario only", "one_severity": "One-axis severity", "one_strategy": "One-axis strategy S0–S3", "two_axis": "Strategy × severity", "three_axis": "Strategy × severity × climate"}[x],
    )
    fixed_strategy = c2.selectbox("Fixed / baseline strategy", list(SCENARIOS.keys()), format_func=lambda x: f"{x} — {SCENARIOS[x]}")
    fixed_severity = c3.selectbox("Fixed severity", list(SEVERITY_LEVELS.keys()), index=1)
    fixed_climate = st.selectbox("Fixed climate", list(CLIMATE_LEVELS.keys()))

    st.info(f"Selected calculation time step: {time_step_hours:g} h. Original daily model is preserved when 24 h is selected. Sub-daily modes use native timestamped CSV/EPW weather when available; daily weather is expanded with a transparent diurnal profile. Energy, pump energy, auxiliary energy, degradation growth, dust accumulation, and cost are scaled by period length.")
    engine_weather_mode, epw_path, csv_path, weather_df, random_state, out_dir = build_weather_controls("scenario")

    include_baseline_layer = st.checkbox("Export baseline no-degradation layer", value=True)
    include_baseline_as_scenario = st.checkbox("Add Baseline Scenario to main output calculation", value=True)
    use_zone_occ = st.checkbox("Use zone-specific occupancy input", value=False)
    zone_df = None
    if use_zone_occ:
        zone_df = st.data_editor(default_zone_table(), num_rows="dynamic", use_container_width=True)

    if st.button("Run selected model", type="primary"):
        try:
            if engine_weather_mode == "uploaded" and weather_df is None:
                st.warning("Upload a CSV/EPW weather file first, or select another weather source.")
                st.stop()
            result = run_scenario_model(
                output_dir=out_dir,
                axis_mode=axis_mode,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode,
                epw_path=epw_path if epw_path else None,
                csv_path=csv_path if csv_path else None,
                weather_df=weather_df,
                fixed_strategy=fixed_strategy,
                fixed_severity=fixed_severity,
                fixed_climate=fixed_climate,
                zone_df=zone_df,
                random_state=random_state,
                include_baseline_layer=include_baseline_layer,
                include_baseline_as_scenario=include_baseline_as_scenario,
                degradation_model=st.session_state.get("degradation_model", "physics"),
                time_step_hours=time_step_hours,
                operation_schedule_df=st.session_state.get("operation_schedule_df") if bool(st.session_state.get("EMS_CUSTOM_SCHEDULE_ENABLED", False)) else None,
            )
            st.session_state["last_result"] = result
            st.session_state["last_result_dir"] = out_dir
            st.session_state["last_zone_df"] = zone_df
            tables = build_detailed_tables(out_dir, bldg=bldg, cfg=cfg, zone_df=zone_df)
            detailed_paths = save_detailed_outputs(out_dir, tables)
            st.session_state["last_detailed_paths"] = detailed_paths
            st.success("Model run and detailed outputs finished.")
            st.json({**result, "extra_detailed_outputs": detailed_paths})
            summary_path = Path(result["summary_csv"])
            if summary_path.exists():
                st.dataframe(pd.read_csv(summary_path), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[6]:
    st.subheader("Early benchmark sensitivity and robustness analysis")
    st.markdown(
        """
        **Early benchmark sensitivity** runs a fast one-at-a-time perturbation around the current setup and ranks parameters by dimensionless elasticity against key KPIs.  
        **Robustness analysis** runs bounded Monte-Carlo input perturbations and reports KPI spread, coefficient of variation, and 5–95% bands.
        """
    )
    c1, c2, c3 = st.columns(3)
    analysis_years = int(c1.number_input("Analysis years", min_value=1, max_value=10, value=1, step=1))
    sens_pct = float(c2.number_input("Sensitivity perturbation ±%", min_value=1.0, max_value=50.0, value=10.0, step=1.0)) / 100.0
    robust_pct = float(c3.number_input("Robustness uncertainty ±%", min_value=1.0, max_value=50.0, value=10.0, step=1.0)) / 100.0
    c1, c2, c3, c4 = st.columns(4)
    sens_strategy = c1.selectbox("Strategy for analysis", list(SCENARIOS.keys()), index=2, key="sens_strategy")
    sens_severity = c2.selectbox("Severity for analysis", list(SEVERITY_LEVELS.keys()), index=1, key="sens_severity")
    sens_climate = c3.selectbox("Climate for analysis", list(CLIMATE_LEVELS.keys()), key="sens_climate")
    n_samples = int(c4.number_input("Robustness samples", min_value=3, max_value=200, value=12, step=1))
    engine_weather_mode_s, epw_path_s, csv_path_s, weather_df_s, random_state_s, out_dir_s = build_weather_controls("sensitivity")
    out_dir_s = str(Path(out_dir_s) / "sensitivity_robustness")
    st.caption(f"Outputs will be written to: {out_dir_s}")

    selected_params = st.multiselect(
        "Optional: limit parameters; leave empty to screen all supported parameters",
        ["conditioned_area_m2", "occupancy_density_p_m2", "lighting_w_m2", "equipment_w_m2", "airflow_m3h_m2", "cooling_intensity_w_m2", "heating_intensity_w_m2", "wall_u", "roof_u", "window_u", "shgc", "glazing_ratio", "infiltration_ach", "COP_COOL_NOM", "COP_HEAT_NOM", "FAN_EFF", "COP_AGING_RATE", "RF_STAR", "B_FOUL", "DUST_RATE", "K_CLOG"],
        default=[],
    )
    col_a, col_b = st.columns(2)
    if col_a.button("Run early benchmark sensitivity", type="primary"):
        try:
            if engine_weather_mode_s == "uploaded" and weather_df_s is None:
                st.warning("Upload weather first, or select synthetic/path weather.")
                st.stop()
            result = run_early_sensitivity_analysis(
                output_dir=out_dir_s,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode_s,
                epw_path=epw_path_s,
                csv_path=csv_path_s,
                weather_df=weather_df_s,
                fixed_strategy=sens_strategy,
                fixed_severity=sens_severity,
                fixed_climate=sens_climate,
                degradation_model=st.session_state.get("degradation_model", "physics"),
                perturbation_pct=sens_pct,
                analysis_years=analysis_years,
                random_state=random_state_s,
                time_step_hours=time_step_hours,
                parameter_names=selected_params or None,
            )
            st.success("Early sensitivity analysis finished.")
            st.json(result)
            df = pd.read_csv(result["ranking_csv"])
            st.dataframe(df, use_container_width=True)
            if not df.empty:
                st.bar_chart(df.set_index("label")["composite_importance"])
        except Exception as e:
            st.exception(e)
    if col_b.button("Run robustness analysis"):
        try:
            if engine_weather_mode_s == "uploaded" and weather_df_s is None:
                st.warning("Upload weather first, or select synthetic/path weather.")
                st.stop()
            result = run_robustness_analysis(
                output_dir=out_dir_s,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode_s,
                epw_path=epw_path_s,
                csv_path=csv_path_s,
                weather_df=weather_df_s,
                fixed_strategy=sens_strategy,
                fixed_severity=sens_severity,
                fixed_climate=sens_climate,
                degradation_model=st.session_state.get("degradation_model", "physics"),
                n_samples=n_samples,
                uncertainty_pct=robust_pct,
                analysis_years=analysis_years,
                random_state=random_state_s,
                time_step_hours=time_step_hours,
                parameter_names=selected_params or None,
            )
            st.success("Robustness analysis finished.")
            st.json(result)
            df = pd.read_csv(result["summary_csv"])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[7]:
    st.subheader("Extra UI tools: validation, benchmark summary, zone tables, upload handling")
    target_folder = st.text_input("Result folder for extra tools", st.session_state.get("last_result_dir", "scenario_run"), key="extra_folder")
    paths = find_result_paths(target_folder)
    if paths["summary"].exists():
        summary_df = pd.read_csv(paths["summary"])
        st.markdown("### Validation upload")
        vfile = st.file_uploader("Upload validation CSV from DesignBuilder, EnergyPlus, measured data, or published reference", type=["csv"], key="validation_file")
        if vfile is not None:
            validation_df = load_validation_file(vfile)
            comparison = build_validation_comparison(summary_df, validation_df, source_name=Path(vfile.name).stem)
            comparison_path = Path(target_folder) / "validation_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            st.dataframe(comparison, use_container_width=True)
            download_file_button(comparison_path, "Download validation_comparison.csv")

        st.markdown("### Benchmark / sensitivity summary sheet")
        if (Path(target_folder) / "benchmark_summary.csv").exists():
            bench = pd.read_csv(Path(target_folder) / "benchmark_summary.csv")
        else:
            tables = build_detailed_tables(target_folder, bldg=bldg, cfg=cfg, zone_df=st.session_state.get("last_zone_df"))
            save_detailed_outputs(target_folder, tables)
            bench = tables.get("benchmark_summary", pd.DataFrame())
        st.dataframe(bench, use_container_width=True)
        if len(bench) and "energy_delta_pct" in bench.columns and "scenario_combo_3axis" in bench.columns:
            st.bar_chart(bench.set_index("scenario_combo_3axis")["energy_delta_pct"])

        st.markdown("### Zone analysis")
        zone_path = Path(target_folder) / "zone_analysis.csv"
        if zone_path.exists():
            zdf = pd.read_csv(zone_path)
            st.dataframe(zdf.head(300), use_container_width=True)
            download_file_button(zone_path, "Download zone_analysis.csv")
        else:
            st.info("Run the model with zone-specific occupancy enabled to generate zone analysis.")
    else:
        st.info("Run a model first, or type an existing result folder.")

with tabs[8]:
    st.subheader("KPI charts")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "scenario_run"), key="kpi_folder"))
    if folder.exists():
        paths = find_result_paths(folder)
        if paths["summary"].exists():
            kpi = pd.read_csv(paths["summary"])
            st.dataframe(kpi, use_container_width=True)
            for metric in ["Total Energy MWh", "Mean Degradation Index", "Mean Comfort Deviation C", "Total CO2 tonne"]:
                if metric in kpi.columns and "scenario_combo_3axis" in kpi.columns:
                    st.line_chart(kpi.set_index("scenario_combo_3axis")[metric])
        figs = folder / "figures"
        if figs.exists():
            img_files = sorted(figs.glob("*.png"))[:24]
            cols = st.columns(2)
            for i, img in enumerate(img_files):
                with cols[i % 2]:
                    st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No result folder found yet.")

with tabs[9]:
    st.subheader("Train CatBoost surrogate")
    dataset_path = st.text_input("Input dataset CSV", str(Path(st.session_state.get("last_result_dir", "scenario_run")) / "matrix_ml_dataset.csv"))
    surrogate_out = st.text_input("Surrogate output folder", "surrogate_run")
    n_iter_search = int(st.number_input("CatBoost search iterations", min_value=2, value=6, step=1))
    shap_sample = int(st.number_input("SHAP sample size", min_value=100, value=1000, step=100))
    if st.button("Train CatBoost surrogate"):
        try:
            result = train_surrogate_models(dataset_path, surrogate_out, n_iter_search, shap_sample, int(42))
            st.success("Surrogate training finished.")
            st.json(result)
            p = Path(result["metrics_csv"])
            if p.exists():
                st.dataframe(pd.read_csv(p), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[10]:
    st.subheader("Exports and results")
    folder = Path(st.text_input("Folder to inspect/export", st.session_state.get("last_result_dir", "scenario_run"), key="export_folder"))
    if folder.exists():
        csvs = sorted(folder.glob("*.csv"))
        st.write(f"CSV files found: {len(csvs)}")
        for csvf in csvs[:18]:
            with st.expander(csvf.name):
                try:
                    st.dataframe(pd.read_csv(csvf).head(80), use_container_width=True)
                except Exception as e:
                    st.warning(str(e))
                download_file_button(csvf, f"Download {csvf.name}", key=f"download_{csvf.name}")
        for special in ["results_export.xlsx", "detailed_outputs.xlsx", "results_report.pdf", "surrogate_export.xlsx", "surrogate_report.pdf"]:
            download_file_button(folder / special, f"Download {special}", key=f"download_{special}")
        if st.button("Create ZIP bundle for this run"):
            zip_path = create_zip_from_folder(folder)
            st.success(f"ZIP created: {zip_path}")
            download_file_button(zip_path, "Download ZIP bundle")
    else:
        st.info("No folder found yet.")

with tabs[11]:
    st.subheader("Model and deployment guide")
    st.markdown(
        """
        ### Calculation basis
        The engine remains the single calculation authority. The Streamlit app only collects inputs, calls `run_scenario_model()`, and displays/exports results.

        ### Time-series selector
        The calculation time-step selector supports Daily, 12-hour, 6-hour, 3-hour, and Hourly. The original model is recovered at 24 h. Sub-daily modes preserve the same load, COP, maintenance, and degradation equations, while scaling duration-dependent terms by the selected period.

        ### Early benchmark sensitivity
        The early sensitivity analysis perturbs each selected parameter down and up around the baseline and ranks parameters by central elasticity:

        `elasticity = (% KPI change) / (% input change)`

        A high absolute elasticity means that the KPI is more sensitive to that input.

        ### Robustness analysis
        Robustness analysis samples uncertain inputs inside a bounded uniform range, runs the selected scenario repeatedly, and reports mean, standard deviation, coefficient of variation, and 5–95% KPI bands.

        ### EMS, schedule, and multi-objective tabs
        The EMS tab adds occupancy reset, night setback, demand response, economizer/free cooling, optimum start, and custom scheduled operation. The Operation Scheduling tab can pass an editable schedule to the engine. The Multi-Objective Optimization tab screens control candidates and marks Pareto candidates across energy, degradation, comfort, and carbon KPIs.

        ### Run locally
        ```bash
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```
        """
    )


with tabs[12]:
    st.subheader("Formal model validation and calibration metrics")
    st.markdown("Upload measured, EnergyPlus, DesignBuilder, or published reference data and calculate formal validation metrics. This tab is independent from the main scenario run and does not alter the model equations.")
    folder = Path(st.text_input("Model result folder", st.session_state.get("last_result_dir", "scenario_run"), key="validation_metric_folder"))
    paths = find_result_paths(folder)
    model_df = pd.DataFrame()
    if paths["summary"].exists():
        model_source = st.selectbox("Model data source", ["summary", "annual", "daily/time-step"], key="validation_model_source")
        model_path = {"summary": paths["summary"], "annual": paths["annual"], "daily/time-step": paths["daily"]}[model_source]
        model_df = pd.read_csv(model_path)
        st.caption(f"Loaded model file: {model_path.name}")
        st.dataframe(model_df.head(80), use_container_width=True)
    else:
        st.info("Run the model first or choose an existing result folder.")
    ref_upload = st.file_uploader("Upload reference CSV for formal validation", type=["csv"], key="formal_validation_upload")
    if not model_df.empty and ref_upload is not None:
        ref_df = load_validation_file(ref_upload)
        st.markdown("### Reference data preview")
        st.dataframe(ref_df.head(80), use_container_width=True)
        num_model_cols = [c for c in model_df.columns if pd.api.types.is_numeric_dtype(model_df[c])]
        num_ref_cols = [c for c in ref_df.columns if pd.api.types.is_numeric_dtype(ref_df[c])]
        c1, c2 = st.columns(2)
        model_col = c1.selectbox("Model KPI column", num_model_cols, index=num_model_cols.index("Total Energy MWh") if "Total Energy MWh" in num_model_cols else 0)
        ref_col = c2.selectbox("Reference KPI column", num_ref_cols)
        if st.button("Calculate validation metrics"):
            metrics = build_formal_validation_metrics(model_df, ref_df, model_col, ref_col)
            out_path = folder / "formal_validation_metrics.csv"
            metrics.to_csv(out_path, index=False)
            st.dataframe(metrics, use_container_width=True)
            download_file_button(out_path, "Download formal_validation_metrics.csv")
            st.info("Calibration interpretation: lower CVRMSE/NMBE and higher R² indicate better agreement. Use this tab to document model validity against measured/simulation reference data.")

with tabs[13]:
    st.subheader("Heat Exchanger Diagnostics")
    st.markdown("Calculate air-side and water-side pressure drops, inlet/outlet temperatures, degraded UA, LMTD, and detailed pump implication from the existing time-step results.")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "scenario_run"), key="hx_folder"))
    paths = find_result_paths(folder)
    if paths["daily"].exists():
        daily_df = pd.read_csv(paths["daily"])
        c1, c2, c3, c4 = st.columns(4)
        air_mode = c1.selectbox("Air inlet mode", ["mixed_air_estimate", "ambient", "fixed"], key="hx_air_mode")
        fixed_air = c2.number_input("Fixed air inlet temperature (°C)", value=26.0, step=0.5, key="hx_fixed_air")
        chw_in = c3.number_input("Chilled-water inlet (°C)", value=7.0, step=0.5, key="hx_chw_in")
        hw_in = c4.number_input("Hot-water inlet (°C)", value=60.0, step=1.0, key="hx_hw_in")
        c1, c2, c3, c4 = st.columns(4)
        water_flow = c1.number_input("Water flow (m³/h); 0 = auto from load", value=0.0, min_value=0.0, step=0.5, key="hx_water_flow")
        water_dp = c2.number_input("Clean water pressure drop (kPa)", value=45.0, min_value=0.0, step=5.0, key="hx_water_dp")
        pump_eff = c3.number_input("Pump efficiency", value=0.65, min_value=0.05, max_value=0.95, step=0.01, key="hx_pump_eff")
        ua_clean = c4.number_input("UA clean (kW/K)", value=120.0, min_value=1.0, step=5.0, key="hx_ua_clean")
        c1, c2, c3 = st.columns(3)
        ua_loss = c1.number_input("UA loss factor at full degradation", value=0.30, min_value=0.0, max_value=0.95, step=0.05, key="hx_ua_loss")
        air_dp_f = c2.number_input("Air-side fouling ΔP factor", value=0.75, min_value=0.0, step=0.05, key="hx_air_dp_f")
        water_dp_f = c3.number_input("Water-side fouling ΔP factor", value=0.35, min_value=0.0, step=0.05, key="hx_water_dp_f")
        if st.button("Build heat-exchanger diagnostics", type="primary"):
            hx = build_heat_exchanger_diagnostics(daily_df, bldg=bldg, cfg=cfg, air_inlet_mode=air_mode, fixed_air_inlet_c=fixed_air, chilled_water_in_c=chw_in, hot_water_in_c=hw_in, water_flow_m3h=water_flow, water_dp_clean_kpa=water_dp, pump_efficiency=pump_eff, ua_clean_kw_k=ua_clean, ua_loss_factor=ua_loss, air_fouling_dp_factor=air_dp_f, water_fouling_dp_factor=water_dp_f)
            out_path = folder / "heat_exchanger_diagnostics.csv"
            hx.to_csv(out_path, index=False)
            st.dataframe(hx.head(500), use_container_width=True)
            download_file_button(out_path, "Download heat_exchanger_diagnostics.csv")
            if PLOTLY_AVAILABLE and not hx.empty:
                fig = px.scatter(hx, x="Q_HVAC_kw", y="dP_air_Pa", color="mode" if "mode" in hx.columns else None, title="HX air-side pressure drop vs load")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily/time-step result file found. Run a scenario first.")

with tabs[14]:
    st.subheader("Part-Load COP Curves")
    st.markdown("Evaluate COP correction using part-load-ratio curves. This creates an alternative publication diagnostic of how PLR can affect thermal HVAC energy.")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "scenario_run"), key="plr_folder"))
    paths = find_result_paths(folder)
    if paths["daily"].exists():
        daily_df = pd.read_csv(paths["daily"])
        c1, c2, c3, c4 = st.columns(4)
        curve_type = c1.selectbox("Curve type", ["linear", "quadratic", "cubic"], index=1, key="plr_curve_type")
        a = c2.number_input("Coefficient a", value=0.85, step=0.05, key="plr_a")
        b = c3.number_input("Coefficient b", value=0.25, step=0.05, key="plr_b")
        c = c4.number_input("Coefficient c", value=-0.10, step=0.05, key="plr_c")
        c1, c2, c3 = st.columns(3)
        d = c1.number_input("Coefficient d", value=0.0, step=0.05, key="plr_d")
        min_mod = c2.number_input("Minimum modifier", value=0.50, min_value=0.10, step=0.05, key="plr_min")
        max_mod = c3.number_input("Maximum modifier", value=1.20, min_value=0.10, step=0.05, key="plr_max")
        if st.button("Build part-load COP analysis", type="primary"):
            plr = build_part_load_curve_analysis(daily_df, cfg=cfg, curve_type=curve_type, coeff_a=a, coeff_b=b, coeff_c=c, coeff_d=d, min_modifier=min_mod, max_modifier=max_mod)
            out_path = folder / "part_load_cop_analysis.csv"
            plr.to_csv(out_path, index=False)
            st.dataframe(plr.head(500), use_container_width=True)
            download_file_button(out_path, "Download part_load_cop_analysis.csv")
            if PLOTLY_AVAILABLE and not plr.empty:
                fig = px.scatter(plr, x="PLR", y="COP_with_PLR", color="mode" if "mode" in plr.columns else None, title="Part-load COP correction")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily/time-step result file found. Run a scenario first.")

with tabs[15]:
    st.subheader("Latent Cooling Load")
    st.markdown("Estimate latent cooling load from outdoor humidity ratio, indoor humidity target, ventilation/infiltration flow, and COP. This is important for warm-humid climates.")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "scenario_run"), key="latent_folder"))
    paths = find_result_paths(folder)
    if paths["daily"].exists():
        daily_df = pd.read_csv(paths["daily"])
        c1, c2, c3, c4 = st.columns(4)
        indoor_rh = c1.number_input("Indoor RH target (%)", value=50.0, min_value=20.0, max_value=80.0, step=1.0, key="latent_indoor_rh")
        indoor_temp = c2.number_input("Indoor temperature for humidity calc (°C)", value=float(st.session_state.get("t_set", 23.0)), step=0.5, key="latent_indoor_temp")
        vent_frac = c3.number_input("Ventilation fraction", value=1.0, min_value=0.0, max_value=2.0, step=0.1, key="latent_vent_frac")
        floor_h = c4.number_input("Floor-to-floor height for infiltration volume (m)", value=3.2, min_value=2.0, max_value=6.0, step=0.1, key="latent_floor_h")
        include_infil = st.checkbox("Include infiltration in latent calculation", value=True, key="latent_infil")
        if st.button("Build latent-load analysis", type="primary"):
            latent = build_latent_load_analysis(daily_df, bldg=bldg, cfg=cfg, indoor_rh_pct=indoor_rh, indoor_temp_c=indoor_temp, ventilation_fraction=vent_frac, include_infiltration=include_infil, floor_to_floor_m=floor_h)
            out_path = folder / "latent_cooling_analysis.csv"
            latent.to_csv(out_path, index=False)
            st.dataframe(latent.head(500), use_container_width=True)
            download_file_button(out_path, "Download latent_cooling_analysis.csv")
            if PLOTLY_AVAILABLE and not latent.empty:
                fig = px.line(latent.head(1000), x="day" if "day" in latent.columns else latent.index, y=["sensible_cooling_kw_model", "latent_cooling_kw", "total_cooling_with_latent_kw"], title="Sensible and latent cooling load")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily/time-step result file found. Run a scenario first.")

with tabs[16]:
    st.subheader("Native Zone-Level Load Analysis")
    st.markdown("Build a stronger zone-level table using zone area, occupancy density, scenario time-step outputs, and load shares. This is a reduced-order zone module for publication reporting.")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "scenario_run"), key="zone_native_folder"))
    paths = find_result_paths(folder)
    zdf_default = st.session_state.get("last_zone_df")
    if zdf_default is None:
        zdf_default = default_zone_table()
    zone_input = st.data_editor(zdf_default, num_rows="dynamic", use_container_width=True, key="native_zone_editor")
    if paths["daily"].exists():
        daily_df = pd.read_csv(paths["daily"])
        if st.button("Build native zone-level load table", type="primary"):
            zloads = build_native_zone_load_table(daily_df, zone_input, bldg=bldg)
            out_path = folder / "native_zone_loads.csv"
            zloads.to_csv(out_path, index=False)
            st.dataframe(zloads.head(500), use_container_width=True)
            download_file_button(out_path, "Download native_zone_loads.csv")
            if PLOTLY_AVAILABLE and not zloads.empty:
                agg = zloads.groupby("zone_name", as_index=False)["zone_energy_kwh_period"].sum()
                fig = px.bar(agg, x="zone_name", y="zone_energy_kwh_period", title="Total energy by zone")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily/time-step result file found. Run a scenario first.")

with tabs[17]:
    st.subheader("Global Sensitivity from Robustness Samples")
    st.markdown("This tab computes a lightweight global-screening index from robustness samples using Pearson and Spearman correlations between uncertain inputs and KPIs.")
    folder = Path(st.text_input("Robustness result folder", str(Path(st.session_state.get("last_result_dir", "scenario_run")) / "sensitivity_robustness"), key="global_sens_folder"))
    samples_path = folder / "robustness_samples.csv"
    upload_samples = st.file_uploader("Or upload robustness_samples.csv", type=["csv"], key="global_sens_upload")
    if upload_samples is not None:
        samples = pd.read_csv(upload_samples)
    elif samples_path.exists():
        samples = pd.read_csv(samples_path)
    else:
        samples = pd.DataFrame()
    if not samples.empty:
        st.dataframe(samples.head(100), use_container_width=True)
        if st.button("Calculate global sensitivity screening", type="primary"):
            gs = build_global_sensitivity_from_samples(samples)
            out_path = folder / "global_sensitivity_screening.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            gs.to_csv(out_path, index=False)
            st.dataframe(gs, use_container_width=True)
            download_file_button(out_path, "Download global_sensitivity_screening.csv")
            if PLOTLY_AVAILABLE and not gs.empty:
                top = gs.head(30)
                fig = px.bar(top, x="importance", y="input_parameter", color="kpi", orientation="h", title="Global sensitivity screening: top input-KPI correlations")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run robustness analysis first, or upload robustness_samples.csv.")

with tabs[18]:
    st.subheader("Advanced Plot Studio")
    st.markdown("Create publication-oriented plots from any CSV output. Supports line, scatter, bar, heatmap, multi-axis line, and combined line+bar charts.")
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Install dependencies from requirements.txt to use Advanced Plot Studio.")
    folder = Path(st.text_input("Folder containing CSV outputs", st.session_state.get("last_result_dir", "scenario_run"), key="plot_folder"))
    csv_files = sorted(folder.glob("*.csv")) if folder.exists() else []
    uploaded_plot_csv = st.file_uploader("Or upload any CSV for plotting", type=["csv"], key="plot_upload")
    plot_df = pd.DataFrame()
    source_name = ""
    if uploaded_plot_csv is not None:
        plot_df = pd.read_csv(uploaded_plot_csv)
        source_name = uploaded_plot_csv.name
    elif csv_files:
        csv_name = st.selectbox("Select CSV", [p.name for p in csv_files], key="plot_csv_select")
        selected_csv = folder / csv_name
        plot_df = pd.read_csv(selected_csv)
        source_name = csv_name
    if not plot_df.empty:
        st.caption(f"Plot source: {source_name} | rows={len(plot_df)}, columns={len(plot_df.columns)}")
        st.dataframe(plot_df.head(100), use_container_width=True)
        cols_all = list(plot_df.columns)
        num_cols = [c for c in cols_all if pd.api.types.is_numeric_dtype(plot_df[c])]
        cat_cols = [c for c in cols_all if c not in num_cols]
        c1, c2, c3 = st.columns(3)
        chart_type = c1.selectbox("Chart type", ["Line", "Scatter", "Bar", "Heatmap", "Multi-axis line", "Combined line + bar"], key="chart_type")
        x_col = c2.selectbox("X-axis", cols_all, index=cols_all.index("day") if "day" in cols_all else 0, key="plot_x")
        group_col = c3.selectbox("Group/color column", ["None"] + cols_all, key="plot_group")
        y_cols = st.multiselect("Y-axis column(s)", num_cols, default=num_cols[:1], key="plot_y_cols")
        title = st.text_input("Chart title", f"{chart_type} from {source_name}", key="plot_title")
        max_rows = int(st.number_input("Max rows to plot", min_value=100, max_value=200000, value=min(10000, max(100, len(plot_df))), step=100, key="plot_max_rows"))
        data = plot_df.head(max_rows).copy()
        if PLOTLY_AVAILABLE and y_cols:
            color_arg = None if group_col == "None" else group_col
            if chart_type == "Line":
                fig = px.line(data, x=x_col, y=y_cols, color=color_arg, title=title)
            elif chart_type == "Scatter":
                fig = px.scatter(data, x=x_col, y=y_cols[0], color=color_arg, title=title)
            elif chart_type == "Bar":
                if color_arg and len(y_cols) == 1:
                    fig = px.bar(data, x=x_col, y=y_cols[0], color=color_arg, title=title, barmode="group")
                else:
                    fig = px.bar(data, x=x_col, y=y_cols, title=title, barmode="group")
            elif chart_type == "Heatmap":
                c1, c2, c3 = st.columns(3)
                y_heat = c1.selectbox("Heatmap Y/category", cols_all, index=cols_all.index(group_col) if group_col in cols_all else 0, key="heat_y")
                value_col = c2.selectbox("Heatmap value", num_cols, index=num_cols.index(y_cols[0]) if y_cols[0] in num_cols else 0, key="heat_val")
                agg_func = c3.selectbox("Aggregation", ["mean", "sum", "median", "max", "min"], key="heat_agg")
                pivot = data.pivot_table(index=y_heat, columns=x_col, values=value_col, aggfunc=agg_func)
                fig = px.imshow(pivot, aspect="auto", title=title, labels=dict(color=value_col))
            elif chart_type == "Multi-axis line":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data[x_col], y=data[y_cols[0]], name=y_cols[0], mode="lines"))
                if len(y_cols) > 1:
                    fig.add_trace(go.Scatter(x=data[x_col], y=data[y_cols[1]], name=y_cols[1], mode="lines", yaxis="y2"))
                    for y in y_cols[2:]:
                        fig.add_trace(go.Scatter(x=data[x_col], y=data[y], name=y, mode="lines"))
                    fig.update_layout(yaxis2=dict(title=y_cols[1], overlaying="y", side="right"))
                fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_cols[0])
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=data[x_col], y=data[y_cols[0]], name=y_cols[0]))
                if len(y_cols) > 1:
                    for y in y_cols[1:]:
                        fig.add_trace(go.Scatter(x=data[x_col], y=data[y], name=y, mode="lines", yaxis="y2"))
                    fig.update_layout(yaxis2=dict(title=", ".join(y_cols[1:]), overlaying="y", side="right"))
                fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_cols[0])
            fig.update_layout(legend_title_text=group_col if group_col != "None" else "Series", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            html = fig.to_html(include_plotlyjs="cdn")
            st.download_button("Download chart HTML", html, file_name="advanced_plot.html", mime="text/html")
        elif y_cols:
            st.line_chart(data.set_index(x_col)[y_cols])
    else:
        st.info("Select a result folder with CSV files, or upload a CSV.")


with tabs[19]:
    st.subheader("Advanced HVAC Control Library")
    st.markdown(
        """
        This tab adds a practical control-strategy library for HVAC research and publication. 
        The implemented rows generate deployable EMS/schedule recommendations. MPC and reinforcement learning are intentionally marked experimental because they require forecasts, training data, and validation before live control.
        """
    )

    st.markdown("### 1. Control-library screening")
    candidates = build_advanced_control_candidates()
    c1, c2, c3, c4, c5 = st.columns(5)
    w_e = c1.number_input("Energy weight", 0.0, 1.0, 0.35, 0.05, key="ctl_w_energy")
    w_c = c2.number_input("Comfort weight", 0.0, 1.0, 0.25, 0.05, key="ctl_w_comfort")
    w_d = c3.number_input("Degradation weight", 0.0, 1.0, 0.20, 0.05, key="ctl_w_degradation")
    w_co = c4.number_input("Carbon weight", 0.0, 1.0, 0.10, 0.05, key="ctl_w_carbon")
    w_f = c5.number_input("Fault-risk weight", 0.0, 1.0, 0.10, 0.05, key="ctl_w_fault")
    scored = build_control_objective_table(
        candidates,
        weights={"energy": w_e, "comfort": w_c, "degradation": w_d, "carbon": w_co, "fault_risk": w_f},
    )
    st.dataframe(scored, use_container_width=True)
    st.download_button(
        "Download advanced_control_library.csv",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="advanced_control_library.csv",
        mime="text/csv",
    )

    st.markdown("### 2. Apply a selected control template to EMS settings")
    control_ids = scored["control_id"].tolist()
    selected_control = st.selectbox("Control template", control_ids, format_func=lambda x: scored.loc[scored["control_id"] == x, "control_name"].iloc[0], key="selected_advanced_control")
    row = scored.loc[scored["control_id"] == selected_control].iloc[0].to_dict()
    st.info(row.get("description", ""))
    c1, c2, c3 = st.columns(3)
    if c1.button("Apply selected template to EMS controls", type="primary"):
        st.session_state["EMS_MODE"] = row["ems_mode"]
        st.session_state["EMS_OCC_CONTROL"] = bool(row["use_occ_reset"])
        st.session_state["EMS_NIGHT_SETBACK"] = bool(row["use_night_setback"])
        st.session_state["EMS_DEMAND_RESPONSE"] = bool(row["use_demand_response"])
        st.session_state["EMS_ECONOMIZER"] = bool(row["use_economizer"])
        st.session_state["EMS_OPTIMUM_START"] = bool(row["use_optimum_start"])
        st.session_state["EMS_LOW_OCC_SETPOINT_SHIFT_C"] = float(row["setpoint_shift_C"])
        st.session_state["EMS_LOW_OCC_AIRFLOW_FACTOR"] = float(row["airflow_factor"])
        st.session_state["EMS_ECONOMIZER_COOLING_REDUCTION"] = float(row["economizer_reduction"])
        st.success("Template applied. Go to 'EMS Control Strategies' and run Scenario Modeling to calculate official KPIs.")
    if c2.button("Create operation schedule from selected template"):
        sched = build_operation_schedule_template()
        # Apply a transparent schedule adjustment according to selected control.
        af = float(row["airflow_factor"])
        shift = float(row["setpoint_shift_C"])
        sched.loc[sched["occupied"] == 0, "airflow_factor"] = min(af, 0.65)
        sched.loc[sched["occupied"] == 0, "setpoint_shift_C"] = max(shift, 1.0)
        if bool(row["use_demand_response"]):
            dr = pd.DataFrame([{
                "day_type": "weekday",
                "start_hour": 13.0,
                "end_hour": 17.0,
                "occupied": 1,
                "occupancy_multiplier": 0.85,
                "setpoint_shift_C": max(shift, 1.2),
                "airflow_factor": min(af, 0.75),
                "cooling_allowed": 1,
                "heating_allowed": 1,
                "demand_response": 1,
            }])
            sched = pd.concat([sched, dr], ignore_index=True)
        st.session_state["operation_schedule_df"] = validate_operation_schedule(sched)
        st.session_state["EMS_CUSTOM_SCHEDULE_ENABLED"] = True
        st.success("Operation schedule created and custom schedule enabled.")
    if c3.button("Mark as candidate for Multi-Objective Optimization"):
        # Defer widget-key update to the next script run. Directly setting
        # st.session_state["moo_optimizer"] here would fail because the selectbox
        # with key="moo_optimizer" has already been created above.
        st.session_state["pending_moo_optimizer"] = "Custom optimizer label"
        st.session_state["selected_control_candidate"] = selected_control
        st.success("Selected control stored. The optimizer selector will switch to Custom optimizer label on rerun.")
        st.rerun()

    st.markdown("### 3. Controller descriptions and expected KPI effect")
    descriptions = scored[[
        "control_id", "control_name", "status", "estimated_energy_saving_pct", "estimated_comfort_risk_score", 
        "estimated_degradation_risk_score", "estimated_carbon_saving_pct", "fault_adaptiveness_score", "description"
    ]]
    st.dataframe(descriptions, use_container_width=True)

    st.markdown("### 4. Experimental MPC template")
    st.warning("MPC is experimental here: the tab prepares a forecast/control-horizon template. It is not a validated autonomous MPC solver yet.")
    c1, c2 = st.columns(2)
    mpc_horizon = int(c1.number_input("MPC prediction horizon (hours)", 2, 168, 24, 1, key="mpc_horizon"))
    mpc_step = int(c2.number_input("MPC control step (hours)", 1, 24, 1, 1, key="mpc_step"))
    mpc_df = build_mpc_experimental_template(mpc_horizon, mpc_step)
    st.dataframe(mpc_df.head(48), use_container_width=True)
    st.download_button("Download mpc_experimental_template.csv", mpc_df.to_csv(index=False).encode("utf-8"), file_name="mpc_experimental_template.csv", mime="text/csv")
    st.markdown(
        """
        **Meaning of experimental MPC:** MPC needs future weather, future occupancy, electricity price/carbon signals, a prediction model, and an optimizer that solves the best sequence of controls at every step. This package prepares the structure and variables, but does not claim a validated live MPC controller.
        """
    )

    st.markdown("### 5. Experimental reinforcement-learning dataset specification")
    st.warning("Reinforcement learning is experimental here: the tab defines state/action/reward fields for offline RL datasets. It does not train or deploy an RL policy.")
    rl_df = build_rl_experimental_dataset_spec()
    st.dataframe(rl_df, use_container_width=True)
    st.download_button("Download rl_dataset_spec.csv", rl_df.to_csv(index=False).encode("utf-8"), file_name="rl_dataset_spec.csv", mime="text/csv")
    st.markdown(
        """
        **Meaning of experimental RL:** RL requires many simulated or measured episodes, safety constraints, reward design, training, testing, and validation. In this bundle it is included as an export-ready research structure so you can generate datasets from the simulator and train RL externally later.
        """
    )


with tabs[20]:
    st.subheader("Core KPI Impact Dashboard")
    st.markdown(
        """
        This dashboard answers the key question: **what is the effect of each selected module on the official core KPIs?**
        It runs the main engine twice: a baseline case and a modified case. The delta is calculated for Energy, Comfort, Carbon, Degradation, COP, Cost, and component energy.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    impact_years = int(c1.number_input("Impact test years", 1, 20, 1, 1, key="impact_years"))
    impact_strategy = c2.selectbox("Fixed strategy for impact", list(SCENARIOS.keys()), index=list(SCENARIOS.keys()).index(st.session_state.get("impact_strategy", "S3")) if st.session_state.get("impact_strategy", "S3") in SCENARIOS else 3, key="impact_strategy")
    impact_severity = c3.selectbox("Fixed severity", list(SEVERITY_LEVELS.keys()), index=list(SEVERITY_LEVELS.keys()).index(st.session_state.get("impact_severity", "Moderate")) if st.session_state.get("impact_severity", "Moderate") in SEVERITY_LEVELS else 1, key="impact_severity")
    impact_climate = c4.selectbox("Fixed climate", list(CLIMATE_LEVELS.keys()), index=list(CLIMATE_LEVELS.keys()).index(st.session_state.get("impact_climate", "C0_Baseline")) if st.session_state.get("impact_climate", "C0_Baseline") in CLIMATE_LEVELS else 0, key="impact_climate")
    c1, c2 = st.columns(2)
    impact_folder = c1.text_input("Impact output folder", "core_kpi_impact_runs", key="impact_folder")
    impact_random_state = int(c2.number_input("Impact random state", 1, 9999, 42, 1, key="impact_random_state"))

    st.markdown("### Select module families to evaluate through the core solver")
    c1, c2, c3 = st.columns(3)
    test_all_current = c1.checkbox("All-current setup vs original core", value=True, key="impact_all_current")
    test_control_template = c2.checkbox("Selected Control Library template", value=True, key="impact_control_template")
    test_ems_schedule = c3.checkbox("Current EMS + operation schedule", value=True, key="impact_ems_schedule")
    c1, c2, c3 = st.columns(3)
    test_coupled_modules = c1.checkbox("Coupled diagnostic modules", value=True, key="impact_coupled_modules")
    test_physics_switches = c2.checkbox("Parameter switches", value=False, key="impact_physics_switches")
    test_each_coupled = c3.checkbox("Each coupled module separately", value=False, key="impact_each_coupled")

    st.info("For speed, this dashboard uses the selected strategy across severity levels in one-axis severity mode. Increase years only after confirming the setup works.")

    if st.button("Run selected core KPI impact analyses", type="primary"):
        all_results = []
        impact_dir = Path(impact_folder)
        base_original = _disable_ems(_disable_coupled_modules(_set_all_physics_switches(_clone_engine_cfg(cfg, impact_years), True)))
        current_cfg_for_run = _clone_engine_cfg(cfg, impact_years)
        current_schedule = st.session_state.get("operation_schedule_df") if bool(getattr(current_cfg_for_run, "EMS_CUSTOM_SCHEDULE_ENABLED", False)) else None

        with st.spinner("Running core KPI impact simulations..."):
            if test_all_current:
                all_results.append(_run_core_impact_pair(
                    "all_current_vs_original_core", base_original, current_cfg_for_run, impact_dir, impact_years,
                    impact_strategy, impact_severity, impact_climate, random_state=impact_random_state,
                    operation_schedule_for_modified=current_schedule,
                ))

            if test_control_template:
                try:
                    candidates = build_advanced_control_candidates()
                    scored = build_control_objective_table(candidates)
                    sel = st.session_state.get("selected_advanced_control", scored["control_id"].iloc[0])
                    row = scored.loc[scored["control_id"] == sel].iloc[0].to_dict()
                    control_cfg = _apply_control_row_to_cfg(_clone_engine_cfg(base_original, impact_years), row)
                    all_results.append(_run_core_impact_pair(
                        "selected_control_library_template", base_original, control_cfg, impact_dir, impact_years,
                        impact_strategy, impact_severity, impact_climate, random_state=impact_random_state,
                    ))
                except Exception as e:
                    st.warning(f"Control-library impact skipped: {e}")

            if test_ems_schedule:
                ems_base = _disable_ems(_clone_engine_cfg(current_cfg_for_run, impact_years))
                ems_mod = _clone_engine_cfg(current_cfg_for_run, impact_years)
                all_results.append(_run_core_impact_pair(
                    "current_ems_and_schedule", ems_base, ems_mod, impact_dir, impact_years,
                    impact_strategy, impact_severity, impact_climate, random_state=impact_random_state,
                    operation_schedule_for_modified=current_schedule,
                ))

            if test_coupled_modules:
                coupled_base = _disable_coupled_modules(_clone_engine_cfg(current_cfg_for_run, impact_years))
                coupled_mod = _clone_engine_cfg(current_cfg_for_run, impact_years)
                all_results.append(_run_core_impact_pair(
                    "all_coupled_diagnostic_modules", coupled_base, coupled_mod, impact_dir, impact_years,
                    impact_strategy, impact_severity, impact_climate, random_state=impact_random_state,
                    operation_schedule_for_modified=current_schedule,
                ))

            if test_physics_switches:
                phys_base = _set_all_physics_switches(_clone_engine_cfg(current_cfg_for_run, impact_years), True)
                phys_mod = _clone_engine_cfg(current_cfg_for_run, impact_years)
                all_results.append(_run_core_impact_pair(
                    "current_parameter_switches", phys_base, phys_mod, impact_dir, impact_years,
                    impact_strategy, impact_severity, impact_climate, random_state=impact_random_state,
                    operation_schedule_for_modified=current_schedule,
                ))

            if test_each_coupled:
                coupled_attrs = [
                    ("part_load_COP", "APPLY_PART_LOAD_COP_TO_CORE"),
                    ("latent_cooling", "APPLY_LATENT_LOAD_TO_CORE"),
                    ("HX_air_pressure_to_fan", "APPLY_HX_AIR_PRESSURE_TO_FAN"),
                    ("HX_water_pressure_to_pump", "APPLY_HX_WATER_PRESSURE_TO_PUMP"),
                    ("HX_UA_capacity", "APPLY_HX_UA_TO_CAPACITY"),
                    ("native_zone_loads", "APPLY_NATIVE_ZONE_LOADS"),
                ]
                for label, attr in coupled_attrs:
                    single_base = _disable_coupled_modules(_clone_engine_cfg(current_cfg_for_run, impact_years))
                    single_mod = _disable_coupled_modules(_clone_engine_cfg(current_cfg_for_run, impact_years))
                    if hasattr(single_mod, attr):
                        setattr(single_mod, attr, True)
                    all_results.append(_run_core_impact_pair(
                        f"module_{label}", single_base, single_mod, impact_dir, impact_years,
                        impact_strategy, impact_severity, impact_climate, random_state=impact_random_state,
                        operation_schedule_for_modified=current_schedule,
                    ))

        if all_results:
            impact_df = pd.concat(all_results, ignore_index=True)
            impact_df.to_csv(Path(impact_folder) / "core_kpi_impact_all_cases.csv", index=False)
            st.session_state["last_core_kpi_impact_df"] = impact_df
            st.success("Core KPI impact analysis completed.")
            st.dataframe(impact_df, use_container_width=True)
        else:
            st.warning("No impact case selected.")

    impact_df = st.session_state.get("last_core_kpi_impact_df")
    if isinstance(impact_df, pd.DataFrame) and len(impact_df) > 0:
        st.markdown("### Latest impact results")
        st.dataframe(impact_df, use_container_width=True)
        st.download_button("Download core_kpi_impact_all_cases.csv", impact_df.to_csv(index=False).encode("utf-8"), file_name="core_kpi_impact_all_cases.csv", mime="text/csv")
        if PLOTLY_AVAILABLE:
            kpi_focus = st.selectbox("KPI to plot", impact_df["kpi"].unique().tolist(), index=0, key="impact_plot_kpi")
            pdat = impact_df[impact_df["kpi"] == kpi_focus]
            fig = px.bar(pdat, x="impact_case", y="delta_pct", title=f"Core KPI impact on {kpi_focus} (%)", text="delta_pct")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How to interpret this dashboard")
    st.markdown(
        """
        - **Baseline** is the reference engine run for the selected impact case.  
        - **Modified** is the same engine run after applying the selected tab/module settings.  
        - **Delta** = modified − baseline. Negative energy or CO₂ delta means saving.  
        - This tab does not use post-processing estimates; it calls the **core scenario solver** and reads the official summary CSV.  
        - Use this dashboard to report the actual effect of Control Library, EMS, schedules, parameter switches, and coupled diagnostics on official KPIs.
        """
    )


with tabs[21]:
    st.subheader("Live Core Solver Lab — official solver playback")
    st.markdown(
        """
        This tab is **fully combined with the core solver**. It does not use a separate JavaScript/ODE demo model.
        When you click **Build official live dataset**, the app calls the same HVAC engine functions used by Scenario Modeling
        (`_load_base_weather`, `aggregate_zone_occupancy`, and `simulate_combo`). The live panel then plays back the official
        time-step rows so the live curves, KPIs, maintenance events, EMS flags, and coupled-module effects are identical to the solver output.
        """
    )

    st.markdown("### 1. Live solver configuration")
    c1, c2, c3, c4 = st.columns(4)
    live_years = int(c1.number_input("Live simulation years", min_value=1, max_value=20, value=int(st.session_state.get("live_years", 1)), step=1, key="live_years"))
    live_strategy = c2.selectbox("Live strategy", list(SCENARIOS.keys()), index=list(SCENARIOS.keys()).index(st.session_state.get("live_strategy", "S3")) if st.session_state.get("live_strategy", "S3") in SCENARIOS else 3, key="live_strategy")
    live_severity = c3.selectbox("Live severity", list(SEVERITY_LEVELS.keys()), index=list(SEVERITY_LEVELS.keys()).index(st.session_state.get("live_severity", "Moderate")) if st.session_state.get("live_severity", "Moderate") in SEVERITY_LEVELS else 1, key="live_severity")
    live_climate = c4.selectbox("Live climate", list(CLIMATE_LEVELS.keys()), index=list(CLIMATE_LEVELS.keys()).index(st.session_state.get("live_climate", "C0_Baseline")) if st.session_state.get("live_climate", "C0_Baseline") in CLIMATE_LEVELS else 0, key="live_climate")

    live_weather_mode, live_epw_path, live_csv_path, live_weather_df, live_random_state, live_out_dir = build_weather_controls(prefix="live_core")

    c1, c2, c3 = st.columns(3)
    live_use_current_schedule = c1.checkbox("Use current operation schedule", value=bool(st.session_state.get("live_use_current_schedule", True)), key="live_use_current_schedule")
    live_use_zone_table = c2.checkbox("Use current zone table", value=bool(st.session_state.get("live_use_zone_table", True)), key="live_use_zone_table")
    live_trim_rows = int(c3.number_input("Max rows for browser playback", min_value=100, max_value=200000, value=int(st.session_state.get("live_trim_rows", 20000)), step=100, key="live_trim_rows"))

    st.caption("For Streamlit Cloud stability, use 1 year and 1-hour/3-hour/6-hour first. A 20-year hourly run is large and should be exported rather than fully animated in-browser.")

    if st.button("Build official live dataset using core solver", type="primary", key="build_live_core_dataset"):
        try:
            live_cfg = _clone_engine_cfg(cfg, live_years)
            base_weather, weather_meta = _load_base_weather(
                weather_mode=live_weather_mode,
                epw_path=live_epw_path,
                csv_path=live_csv_path,
                weather_df=live_weather_df,
                random_state=live_random_state,
                time_step_hours=getattr(live_cfg, "TIME_STEP_HOURS", 24.0),
            )
            live_zone_df = st.session_state.get("last_zone_df") if live_use_zone_table else None
            live_bldg, live_zone_meta = aggregate_zone_occupancy(bldg, live_zone_df)
            live_schedule_profile = live_zone_meta.get("schedule_profile", None)
            live_schedule_df = None
            if live_use_current_schedule and bool(getattr(live_cfg, "EMS_CUSTOM_SCHEDULE_ENABLED", False)):
                live_schedule_df = st.session_state.get("operation_schedule_df")

            live_daily, live_annual, live_summary = simulate_combo(
                strategy=live_strategy,
                severity=live_severity,
                climate_name=live_climate,
                bldg=live_bldg,
                base_cfg=live_cfg,
                base_weather=base_weather,
                schedule_profile=live_schedule_profile,
                random_state=live_random_state,
                degradation_model=getattr(live_cfg, "degradation_model", "physics"),
                operation_schedule_df=live_schedule_df,
            )
            outp = Path(live_out_dir)
            outp.mkdir(parents=True, exist_ok=True)
            live_daily.to_csv(outp / "live_core_solver_timeseries.csv", index=False)
            live_annual.to_csv(outp / "live_core_solver_annual.csv", index=False)
            pd.DataFrame([live_summary]).to_csv(outp / "live_core_solver_summary.csv", index=False)
            base_weather.to_csv(outp / "live_core_weather_timeseries.csv", index=False)
            with open(outp / "live_core_metadata.json", "w", encoding="utf-8") as f:
                json.dump({"weather_meta": weather_meta, "zone_meta": live_zone_meta, "strategy": live_strategy, "severity": live_severity, "climate": live_climate, "years": live_years, "time_step_hours": getattr(live_cfg, "TIME_STEP_HOURS", 24.0)}, f, indent=2)

            if len(live_daily) > live_trim_rows:
                # Keep the full file on disk but use a copy for browser playback.
                st.info(f"Full live dataset has {len(live_daily):,} rows. Browser playback is limited to the first {live_trim_rows:,} rows; download the full CSV from the output folder.")
                live_playback_df = live_daily.head(live_trim_rows).copy()
            else:
                live_playback_df = live_daily.copy()

            st.session_state["live_core_df"] = live_playback_df
            st.session_state["live_core_full_rows"] = int(len(live_daily))
            st.session_state["live_core_idx"] = 0
            st.session_state["live_core_running"] = False
            st.session_state["live_core_summary"] = live_summary
            st.session_state["live_core_out_dir"] = str(outp)
            st.success("Official live dataset created from the core solver.")
        except Exception as e:
            st.error(f"Live core solver build failed: {e}")

    live_df = st.session_state.get("live_core_df")
    if isinstance(live_df, pd.DataFrame) and len(live_df) > 0:
        st.markdown("### 2. Live playback controls")
        c1, c2, c3, c4, c5 = st.columns(5)
        steps_per_refresh = int(c1.number_input("Steps per refresh", min_value=1, max_value=500, value=int(st.session_state.get("live_steps_per_refresh", 5)), step=1, key="live_steps_per_refresh"))
        refresh_seconds = float(c2.number_input("Refresh seconds", min_value=0.2, max_value=10.0, value=float(st.session_state.get("live_refresh_seconds", 1.0)), step=0.1, key="live_refresh_seconds"))
        if c3.button("Start", key="live_start"):
            st.session_state["live_core_running"] = True
        if c4.button("Stop", key="live_stop"):
            st.session_state["live_core_running"] = False
        if c5.button("Reset", key="live_reset"):
            st.session_state["live_core_idx"] = 0
            st.session_state["live_core_running"] = False

        c1, c2, c3 = st.columns(3)
        if c1.button("Step +1", key="live_step_one"):
            st.session_state["live_core_idx"] = min(int(st.session_state.get("live_core_idx", 0)) + 1, len(live_df) - 1)
        if c2.button("Step +100", key="live_step_100"):
            st.session_state["live_core_idx"] = min(int(st.session_state.get("live_core_idx", 0)) + 100, len(live_df) - 1)
        if c3.button("Jump to end", key="live_jump_end"):
            st.session_state["live_core_idx"] = len(live_df) - 1
            st.session_state["live_core_running"] = False

        current_idx = int(st.session_state.get("live_core_idx", 0))
        current_idx = max(0, min(current_idx, len(live_df) - 1))
        current_idx = int(st.slider("Playback position", min_value=0, max_value=len(live_df) - 1, value=current_idx, step=1, key="live_slider_idx"))
        st.session_state["live_core_idx"] = current_idx

        current = live_df.iloc[current_idx]
        visible = live_df.iloc[: current_idx + 1].copy()

        st.markdown("### 3. Current live state from official solver")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Step", f"{int(current.get('step', current_idx+1)):,}")
        c2.metric("P_total", f"{float(current.get('P_total_kw', 0.0)):.2f} kW")
        c3.metric("COP", f"{float(current.get('COP_eff', 0.0)):.3f}")
        c4.metric("Degradation", f"{float(current.get('delta', 0.0)):.3f}")
        c5.metric("Comfort dev", f"{float(current.get('comfort_dev_C', 0.0)):.2f} °C")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cumulative energy", f"{visible['energy_kwh_period'].sum()/1000.0:.3f} MWh")
        c2.metric("Cumulative CO₂", f"{visible.get('co2_kg_period', pd.Series(dtype=float)).sum()/1000.0:.3f} t")
        c3.metric("HX cleanings", f"{int(visible.get('hx_cleaned', pd.Series(dtype=int)).sum())}")
        c4.metric("Filter repl.", f"{int(visible.get('filter_replaced', pd.Series(dtype=int)).sum())}")
        c5.metric("Mode", str(current.get("mode", "-")))

        # Compact schematic/status strip.
        deg = float(current.get("delta", 0.0))
        cop = float(current.get("COP_eff", 0.0))
        color = "#22c55e" if deg < 0.45 else ("#f59e0b" if deg < 0.70 else "#ef4444")
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.12);border-radius:18px;padding:14px;background:#08111f;margin:10px 0 20px 0;">
              <div style="display:flex;gap:12px;align-items:center;justify-content:space-between;">
                <div style="padding:12px;border-radius:14px;background:#0f1c30;border:1px solid #1f3b63;min-width:160px;text-align:center;">Weather<br><b>{float(current.get('T_amb_C',0)):.1f} °C</b><br>RH {float(current.get('RH_mean_pct',0)):.0f}%</div>
                <div style="font-size:28px;color:#93c5fd;">→</div>
                <div style="padding:12px;border-radius:14px;background:#0f1c30;border:1px solid #1f3b63;min-width:160px;text-align:center;">Load<br><b>{float(current.get('Q_HVAC_kw',0)):.1f} kW</b><br>{str(current.get('mode','-'))}</div>
                <div style="font-size:28px;color:#93c5fd;">→</div>
                <div style="padding:12px;border-radius:14px;background:#0f1c30;border:1px solid {color};min-width:160px;text-align:center;">HVAC Core<br><b>COP {cop:.2f}</b><br>DI {deg:.2f}</div>
                <div style="font-size:28px;color:#93c5fd;">→</div>
                <div style="padding:12px;border-radius:14px;background:#0f1c30;border:1px solid #1f3b63;min-width:160px;text-align:center;">Outputs<br><b>{float(current.get('energy_kwh_period',0)):.1f} kWh</b><br>{float(current.get('co2_kg_period',0)):.1f} kg CO₂</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 4. Live curves")
        if PLOTLY_AVAILABLE:
            plot_cols = [c for c in ["P_total_kw", "COP_eff", "delta", "energy_kwh_period", "comfort_dev_C"] if c in visible.columns]
            fig = go.Figure()
            xcol = "elapsed_days" if "elapsed_days" in visible.columns else ("step" if "step" in visible.columns else visible.index)
            xvals = visible[xcol] if isinstance(xcol, str) else visible.index
            for col in plot_cols:
                fig.add_trace(go.Scatter(x=xvals, y=pd.to_numeric(visible[col], errors="coerce"), mode="lines", name=col))
            maint = visible[(visible.get("hx_cleaned", 0) == 1) | (visible.get("filter_replaced", 0) == 1)] if "hx_cleaned" in visible.columns else pd.DataFrame()
            if not maint.empty:
                fig.add_trace(go.Scatter(x=maint[xcol], y=pd.to_numeric(maint.get("P_total_kw", 0), errors="coerce"), mode="markers", name="maintenance event", marker=dict(size=9, symbol="x")))
            fig.update_layout(template="plotly_dark", height=430, title="Official core solver live playback", legend_orientation="h")
            st.plotly_chart(fig, use_container_width=True)

            comp_cols = [c for c in ["thermal_hvac_kwh_period", "fan_kwh_period", "pump_kwh_period", "auxiliary_kwh_period"] if c in visible.columns]
            if comp_cols:
                comp = pd.DataFrame({"component": comp_cols, "kWh": [float(pd.to_numeric(visible[c], errors="coerce").fillna(0).sum()) for c in comp_cols]})
                fig2 = px.bar(comp, x="component", y="kWh", title="Cumulative component energy during playback")
                fig2.update_layout(template="plotly_dark", height=330)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(visible[[c for c in ["P_total_kw", "COP_eff", "delta", "comfort_dev_C"] if c in visible.columns]])

        st.markdown("### 5. Data and export")
        st.dataframe(visible.tail(200), use_container_width=True)
        st.download_button("Download visible live playback CSV", visible.to_csv(index=False).encode("utf-8"), file_name="live_core_solver_visible_playback.csv", mime="text/csv")
        full_out = Path(st.session_state.get("live_core_out_dir", "")) / "live_core_solver_timeseries.csv"
        if full_out.exists():
            download_file_button(full_out, "Download full live_core_solver_timeseries.csv", key="dl_full_live_core")

        if bool(st.session_state.get("live_core_running", False)):
            next_idx = min(current_idx + steps_per_refresh, len(live_df) - 1)
            st.session_state["live_core_idx"] = int(next_idx)
            if next_idx >= len(live_df) - 1:
                st.session_state["live_core_running"] = False
            time.sleep(refresh_seconds)
            st.rerun()
    else:
        st.info("Build an official live dataset first. The live playback will then animate the exact rows generated by the core HVAC solver.")

    st.markdown("### Why this is stronger than a separate live demo")
    st.markdown(
        """
        - The live dataset is produced by the **same solver** as the Scenario Modeling tab.  
        - EMS settings, operation schedule, parameter switches, latent cooling coupling, part-load COP coupling, HX pressure-drop coupling, HX UA capacity coupling, pump/auxiliary energy, degradation, and maintenance logic are all included.  
        - The live tab is therefore suitable for PhD defence demonstrations because the animated plots are not decorative; they are the official time-step simulation results.  
        - For publication, export `live_core_solver_timeseries.csv` and cite it as a time-resolved solver trace.
        """
    )
