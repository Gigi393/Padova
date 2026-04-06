from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.risk import risk_index
from simglucose.patient.t1dpatient import Action, T1DPatient
from simglucose.sensor.cgm import CGMSensor

BASE_DIR = Path(__file__).resolve().parent
QUEST_FILE = BASE_DIR / "simglucose" / "params" / "Quest.csv"


def _normalize_text(value: str | None, field_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _get_patient_control_params(patient: T1DPatient) -> tuple[float, float]:
    params = patient._params
    basal_rate = params.u2ss * params.BW / 6000.0

    quest = pd.read_csv(QUEST_FILE)
    match = quest.loc[quest["Name"] == patient.name]
    carb_ratio = float(match.iloc[0]["CR"]) if not match.empty else 15.0

    return float(basal_rate), carb_ratio


def run_24h_simulation(
    patient_id: str,
    controller_type: str,
    ctrl_params: dict,
    what_if_condition: str = "baseline",
) -> pd.DataFrame:
    """
    Run a 24-hour minute-resolution simulation and return a report-compatible
    MultiIndex dataframe indexed by (Patient, Time).
    """
    patient_id = _normalize_text(patient_id, "patient_id")
    controller_type = _normalize_text(controller_type, "controller_type").lower()
    what_if_condition = _normalize_text(what_if_condition, "what_if_condition").lower()

    try:
        patient = T1DPatient.withName(patient_id)
    except Exception as exc:
        raise ValueError(f"Unable to load patient '{patient_id}'") from exc

    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")
    basal_rate, carb_ratio = _get_patient_control_params(patient)

    meals = {480: 45.0, 780: 70.0, 1140: 60.0}
    integral_error = 0.0
    prev_error = 0.0
    target_bg = float(ctrl_params.get("target_bg", 110.0))

    results = []
    cgm_history = []
    start_time = datetime(2026, 4, 4, 0, 0, 0)

    supported_conditions = {
        "baseline",
        "carb_overestimate_20",
        "carb_underestimate_20",
        "missed_meal_input",
        "slower_sensor",
    }
    if what_if_condition not in supported_conditions:
        raise ValueError(
            f"Unsupported what_if_condition '{what_if_condition}'. "
            f"Use one of {sorted(supported_conditions)}."
        )

    if controller_type not in {"pid", "bb"}:
        raise ValueError(
            f"Unsupported controller_type '{controller_type}'. Use 'pid' or 'bb'."
        )

    for t in range(1440):
        actual_carbs = meals.get(t, 0.0)
        perceived_carbs = actual_carbs

        if what_if_condition == "carb_overestimate_20":
            perceived_carbs = actual_carbs * 1.2
        elif what_if_condition == "carb_underestimate_20":
            perceived_carbs = actual_carbs * 0.8
        elif what_if_condition == "missed_meal_input":
            perceived_carbs = 0.0

        current_cgm = sensor.measure(patient)
        cgm_history.append(current_cgm)

        if what_if_condition == "slower_sensor" and len(cgm_history) > 15:
            current_cgm = cgm_history[-15]

        requested_basal = basal_rate
        requested_bolus = 0.0

        if controller_type == "pid":
            error = current_cgm - target_bg
            integral_error += error
            derivative_error = error - prev_error
            prev_error = error

            pid_adj = (
                ctrl_params.get("kp", 0.0) * error
                + ctrl_params.get("ki", 0.0) * integral_error
                + ctrl_params.get("kd", 0.0) * derivative_error
            )
            requested_basal = max(0.0, basal_rate + pid_adj)

            if perceived_carbs > 0:
                requested_bolus = perceived_carbs / carb_ratio

        elif controller_type == "bb":
            if perceived_carbs > 0:
                adjusted_cr = carb_ratio * ctrl_params.get("cr_multiplier", 1.0)
                requested_bolus = perceived_carbs / adjusted_cr

        actual_basal = pump.basal(requested_basal)
        actual_bolus = pump.bolus(requested_bolus)
        total_insulin = actual_basal + actual_bolus

        patient.step(Action(insulin=total_insulin, CHO=actual_carbs))

        results.append(
            {
                "Patient": patient_id,
                "Time": start_time + timedelta(minutes=t),
                "BG": patient.observation.Gsub,
                "CGM": current_cgm,
                "CHO": actual_carbs,
                "insulin": total_insulin,
            }
        )

    df = pd.DataFrame(results)
    df.set_index(["Patient", "Time"], inplace=True)
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate optimization-friendly glucose metrics from the BG column."""
    if df.empty:
        return {
            "TIR": 0.0,
            "Hypo": 0.0,
            "Hyper": 0.0,
            "MeanBG": 0.0,
            "StdBG": 0.0,
            "CV": 0.0,
            "LBGI": 0.0,
            "HBGI": 0.0,
            "Risk": 0.0,
            "SafetyScore": 100.0,
            "VariabilityScore": 100.0,
            "CompositeScore": 90.0,
        }

    bgs = df["BG"].to_numpy(dtype=float)
    tir = float(round(np.sum((bgs >= 70) & (bgs <= 180)) / len(bgs) * 100, 2))
    hypo = float(round(np.sum(bgs < 70) / len(bgs) * 100, 2))
    hyper = float(round(np.sum(bgs > 180) / len(bgs) * 100, 2))

    mean_bg = float(round(np.mean(bgs), 2))
    std_bg = float(round(np.std(bgs), 2))
    cv = float(round((std_bg / mean_bg) * 100, 2)) if mean_bg > 0 else 0.0

    lbgi, hbgi, risk = risk_index(bgs.tolist(), len(bgs))
    lbgi = float(round(lbgi, 2))
    hbgi = float(round(hbgi, 2))
    risk = float(round(risk, 2))

    safety_score = float(round(max(0.0, 100.0 - hypo), 2))
    variability_score = float(round(max(0.0, 100.0 - cv), 2))
    composite_score = float(
        round(0.6 * tir + 0.3 * safety_score + 0.1 * variability_score, 2)
    )

    return {
        "TIR": tir,
        "Hypo": hypo,
        "Hyper": hyper,
        "MeanBG": mean_bg,
        "StdBG": std_bg,
        "CV": cv,
        "LBGI": lbgi,
        "HBGI": hbgi,
        "Risk": risk,
        "SafetyScore": safety_score,
        "VariabilityScore": variability_score,
        "CompositeScore": composite_score,
    }
