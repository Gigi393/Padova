import itertools
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from fastmcp import FastMCP

from simglucose.analysis.report import report
from simulation_engine import calculate_metrics, run_24h_simulation

DEFAULT_CONDITIONS = [
    "baseline",
    "slower_sensor",
    "missed_meal_input",
    "carb_overestimate_20",
    "carb_underestimate_20",
]

mcp = FastMCP("EngineeringDesignSimulator")
optimization_history: list[dict] = []


def _normalize_text(value: str | None, field_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_controller_type(controller_type: str) -> str:
    return _normalize_text(controller_type, "controller_type").lower()


def _normalize_patient_id(patient_id: str) -> str:
    return _normalize_text(patient_id, "patient_id")


def _normalize_condition(condition: str) -> str:
    return _normalize_text(condition, "what_if_condition").lower()


def _normalize_conditions(conditions: list[str] | None) -> list[str]:
    source = conditions or DEFAULT_CONDITIONS
    normalized = []
    for condition in source:
        normalized_condition = _normalize_condition(condition)
        if normalized_condition not in normalized:
            normalized.append(normalized_condition)
    return normalized


def _label_run(df: pd.DataFrame, run_label: str) -> pd.DataFrame:
    labeled = df.reset_index()
    labeled["Patient"] = run_label
    labeled = labeled.set_index(["Patient", "Time"])
    return labeled


def _average_metric_dict(metric_rows: list[dict]) -> dict:
    metric_frame = pd.DataFrame(metric_rows)
    return {
        column: round(float(metric_frame[column].mean()), 2)
        for column in metric_frame.columns
        if column != "patient_id"
    }


def _build_run_record(
    patient_id: str,
    controller_type: str,
    params: dict,
    condition: str,
    metrics: dict,
) -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "patient_id": patient_id,
        "controller_type": controller_type,
        "params": dict(params),
        "condition": condition,
        "metrics": metrics,
    }


def _store_run(record: dict) -> None:
    optimization_history.append(record)


def _simulate_and_record(
    patient_id: str,
    controller_type: str,
    params: dict,
    condition: str = "baseline",
) -> tuple[pd.DataFrame, dict]:
    df = run_24h_simulation(patient_id, controller_type, params, condition)
    metrics = calculate_metrics(df)
    _store_run(_build_run_record(patient_id, controller_type, params, condition, metrics))
    return df, metrics


def _candidate_sort_key(candidate: dict) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (
        metrics["CompositeScore"],
        metrics["TIR"],
        -metrics["Hypo"],
    )


def _grid_candidates(controller_type: str, param_ranges: dict) -> list[dict]:
    if controller_type == "pid":
        return [
            {
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "target_bg": target_bg,
            }
            for kp, ki, kd, target_bg in itertools.product(
                param_ranges.get("kp", [0.001]),
                param_ranges.get("ki", [0.0001]),
                param_ranges.get("kd", [0.01]),
                param_ranges.get("target_bg", [110.0]),
            )
        ]

    if controller_type == "bb":
        return [
            {"cr_multiplier": cr_multiplier}
            for cr_multiplier in param_ranges.get("cr_multiplier", [0.8, 1.0, 1.2])
        ]

    raise ValueError(
        f"Unsupported controller_type '{controller_type}'. Use 'pid' or 'bb'."
    )


@mcp.tool()
def tool_run_single_simulation(patient_id: str, controller_type: str, params: dict) -> dict:
    """Run one baseline simulation and return structured optimization metrics."""
    try:
        patient_id = _normalize_patient_id(patient_id)
        controller_type = _normalize_controller_type(controller_type)
        _, metrics = _simulate_and_record(patient_id, controller_type, params, "baseline")
        return {
            "status": "ok",
            "patient_id": patient_id,
            "controller_type": controller_type,
            "params": params,
            "condition": "baseline",
            "metrics": metrics,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_batch_grid_search(
    patient_id: str,
    controller_type: str,
    param_ranges: dict,
    top_n: int = 5,
) -> dict:
    """
    Evaluate a parameter grid on the baseline scenario and return top candidates
    ranked by composite score.
    """
    try:
        patient_id = _normalize_patient_id(patient_id)
        controller_type = _normalize_controller_type(controller_type)
        top_n = max(1, int(top_n))

        candidates = _grid_candidates(controller_type, param_ranges)
        if not candidates:
            raise ValueError("The parameter grid is empty.")

        results = []
        for params in candidates:
            _, metrics = _simulate_and_record(patient_id, controller_type, params, "baseline")
            results.append({"params": params, "metrics": metrics})

        ranked = sorted(results, key=_candidate_sort_key, reverse=True)
        return {
            "status": "ok",
            "patient_id": patient_id,
            "controller_type": controller_type,
            "ranking_metric": "CompositeScore",
            "weights": {"TIR": 0.6, "SafetyScore": 0.3, "VariabilityScore": 0.1},
            "total_tested": len(ranked),
            "best_result": ranked[0],
            "top_results": ranked[:top_n],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_validate_scenarios(
    patient_id: str,
    controller_type: str,
    optimized_params: dict,
    what_if_condition: str,
) -> dict:
    """Compare baseline vs one perturbation scenario and generate report plots."""
    try:
        patient_id = _normalize_patient_id(patient_id)
        controller_type = _normalize_controller_type(controller_type)
        condition = _normalize_condition(what_if_condition)

        df_base, base_metrics = _simulate_and_record(
            patient_id, controller_type, optimized_params, "baseline"
        )
        if condition == "baseline":
            df_pert = df_base
            pert_metrics = base_metrics
            comparison_df = _label_run(df_base, f"{patient_id}|baseline")
        else:
            df_pert, pert_metrics = _simulate_and_record(
                patient_id,
                controller_type,
                optimized_params,
                condition,
            )
            comparison_df = pd.concat(
                [
                    _label_run(df_base, f"{patient_id}|baseline"),
                    _label_run(df_pert, f"{patient_id}|{condition}"),
                ]
            )

        save_path = os.path.abspath(f"results_{patient_id}_{controller_type}_{condition}")
        os.makedirs(save_path, exist_ok=True)
        report(comparison_df, save_path=save_path)

        return {
            "status": "ok",
            "patient_id": patient_id,
            "controller_type": controller_type,
            "params": optimized_params,
            "baseline_metrics": base_metrics,
            "perturbed_metrics": pert_metrics,
            "tir_delta": round(pert_metrics["TIR"] - base_metrics["TIR"], 2),
            "composite_delta": round(
                pert_metrics["CompositeScore"] - base_metrics["CompositeScore"], 2
            ),
            "report_path": save_path,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_validate_condition_suite(
    patient_id: str,
    controller_type: str,
    optimized_params: dict,
    conditions: list[str] | None = None,
) -> dict:
    """
    Run one parameter set across multiple perturbation conditions and return
    aggregate robustness metrics.
    """
    try:
        patient_id = _normalize_patient_id(patient_id)
        controller_type = _normalize_controller_type(controller_type)
        normalized_conditions = _normalize_conditions(conditions)

        per_condition = {}
        for condition in normalized_conditions:
            _, metrics = _simulate_and_record(
                patient_id, controller_type, optimized_params, condition
            )
            per_condition[condition] = metrics

        metric_names = [
            "TIR",
            "Hypo",
            "Hyper",
            "CV",
            "CompositeScore",
            "SafetyScore",
            "VariabilityScore",
        ]
        average_metrics = {
            name: round(
                sum(metrics[name] for metrics in per_condition.values())
                / len(per_condition),
                2,
            )
            for name in metric_names
        }
        worst_tir = min(metrics["TIR"] for metrics in per_condition.values())
        worst_hypo = max(metrics["Hypo"] for metrics in per_condition.values())

        return {
            "status": "ok",
            "patient_id": patient_id,
            "controller_type": controller_type,
            "params": optimized_params,
            "conditions_tested": normalized_conditions,
            "per_condition": per_condition,
            "average_metrics": average_metrics,
            "worst_case_tir": round(worst_tir, 2),
            "worst_case_hypo": round(worst_hypo, 2),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_validate_population_scenario(
    patient_ids: list[str],
    controller_type: str,
    optimized_params: dict,
    what_if_condition: str,
    save_path: str | None = None,
) -> dict:
    """
    Run one scenario across multiple patients, save a combined ensemble report,
    and return aggregate per-patient metrics.
    """
    try:
        if not patient_ids:
            raise ValueError("patient_ids must contain at least one patient.")

        controller_type = _normalize_controller_type(controller_type)
        condition = _normalize_condition(what_if_condition)
        normalized_patient_ids = [_normalize_patient_id(patient_id) for patient_id in patient_ids]

        combined_frames = []
        per_patient_rows = []

        for patient_id in normalized_patient_ids:
            df, metrics = _simulate_and_record(
                patient_id, controller_type, optimized_params, condition
            )
            combined_frames.append(df)
            per_patient_rows.append({"patient_id": patient_id, **metrics})

        combined_df = pd.concat(combined_frames).sort_index()
        patient_metrics_df = pd.DataFrame(per_patient_rows)
        average_metrics = _average_metric_dict(per_patient_rows)
        worst_tir = round(float(patient_metrics_df["TIR"].min()), 2)
        worst_hypo = round(float(patient_metrics_df["Hypo"].max()), 2)

        if save_path is None or not save_path.strip():
            save_path = os.path.abspath(
                f"results_population_{controller_type}_{condition}"
            )
        else:
            save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)

        patient_metrics_df.to_csv(os.path.join(save_path, "patient_metrics.csv"), index=False)
        summary = {
            "controller_type": controller_type,
            "scenario": condition,
            "params": optimized_params,
            "patient_count": len(normalized_patient_ids),
            "average_metrics": average_metrics,
            "worst_tir": worst_tir,
            "worst_hypo": worst_hypo,
            "report_path": save_path,
        }
        with open(os.path.join(save_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        report(combined_df, save_path=save_path)

        return {"status": "ok", **summary}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_optimize_and_validate(
    patient_id: str,
    controller_type: str,
    param_ranges: dict,
    conditions: list[str] | None = None,
    top_n: int = 3,
) -> dict:
    """
    Run baseline grid search, then validate the top candidates across a condition
    suite and return the most robust candidate.
    """
    try:
        patient_id = _normalize_patient_id(patient_id)
        controller_type = _normalize_controller_type(controller_type)
        normalized_conditions = _normalize_conditions(conditions)
        top_n = max(1, int(top_n))

        candidates = _grid_candidates(controller_type, param_ranges)
        baseline_results = []
        for params in candidates:
            _, metrics = _simulate_and_record(patient_id, controller_type, params, "baseline")
            baseline_results.append({"params": params, "metrics": metrics})

        ranked_baseline = sorted(baseline_results, key=_candidate_sort_key, reverse=True)
        finalists = []
        for candidate in ranked_baseline[:top_n]:
            per_condition = {}
            for condition in normalized_conditions:
                _, metrics = _simulate_and_record(
                    patient_id, controller_type, candidate["params"], condition
                )
                per_condition[condition] = metrics

            average_composite = round(
                sum(metrics["CompositeScore"] for metrics in per_condition.values())
                / len(per_condition),
                2,
            )
            worst_tir = round(min(metrics["TIR"] for metrics in per_condition.values()), 2)
            worst_hypo = round(max(metrics["Hypo"] for metrics in per_condition.values()), 2)

            finalists.append(
                {
                    "params": candidate["params"],
                    "baseline_metrics": candidate["metrics"],
                    "validation_metrics": per_condition,
                    "average_composite": average_composite,
                    "worst_tir": worst_tir,
                    "worst_hypo": worst_hypo,
                }
            )

        finalists.sort(
            key=lambda item: (
                item["average_composite"],
                item["worst_tir"],
                -item["worst_hypo"],
            ),
            reverse=True,
        )

        return {
            "status": "ok",
            "patient_id": patient_id,
            "controller_type": controller_type,
            "ranking_metric": "average_composite",
            "conditions_tested": normalized_conditions,
            "best_candidate": finalists[0] if finalists else None,
            "finalists": finalists,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_get_history(
    patient_id: str | None = None,
    controller_type: str | None = None,
    top_n: int = 10,
) -> dict:
    """
    Return stored optimization history, optionally filtered by patient or controller.
    """
    try:
        top_n = max(1, int(top_n))
        records = optimization_history

        if patient_id is not None and patient_id.strip():
            normalized_patient_id = _normalize_patient_id(patient_id)
            records = [record for record in records if record["patient_id"] == normalized_patient_id]

        if controller_type is not None and controller_type.strip():
            normalized_controller = _normalize_controller_type(controller_type)
            records = [
                record for record in records if record["controller_type"] == normalized_controller
            ]

        ranked = sorted(records, key=lambda record: _candidate_sort_key(record), reverse=True)
        return {
            "status": "ok",
            "total_records": len(records),
            "top_records": ranked[:top_n],
            "best_record": ranked[0] if ranked else None,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def tool_reset_history() -> dict:
    """Clear stored optimization history for a fresh optimization session."""
    optimization_history.clear()
    return {"status": "ok", "message": "Optimization history cleared."}


if __name__ == "__main__":
    mcp.run()
