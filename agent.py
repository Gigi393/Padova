import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests

import MCPserver as mcpserver

BASE_DIR = Path(__file__).resolve().parent
PATIENT_FILE = BASE_DIR / "simglucose" / "params" / "vpatient_params.csv"
DEFAULT_SCENARIOS = [
    "baseline",
    "slower_sensor",
    "missed_meal_input",
    "carb_overestimate_20",
    "carb_underestimate_20",
]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPTIMIZATION_BUDGET = 20
PROPOSAL_BATCH_SIZE = 5


@dataclass(frozen=True)
class CandidateResult:
    controller_type: str
    params: dict
    average_metrics: dict
    worst_tir: float
    worst_hypo: float
    patient_count: int


ORIGINAL_REPORT = mcpserver.report


def _patch_mcp_report() -> None:
    def wrapped_report(df, cgm_sensor=None, save_path=None):
        results, ri_per_hour, zone_stats, figs, axes = ORIGINAL_REPORT(
            df, cgm_sensor=cgm_sensor, save_path=save_path
        )

        if save_path and figs:
            ensemble_fig = figs[0]
            for axis in axes[:2]:
                legend = axis.get_legend()
                if legend is not None:
                    legend.remove()
                    axis.legend(
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0.0,
                        fontsize=8,
                    )
            ensemble_fig.set_size_inches(14, 10)
            ensemble_fig.tight_layout(rect=[0, 0, 0.82, 1])
            ensemble_fig.savefig(
                Path(save_path) / "BG_trace.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(ensemble_fig)
            plt.close("all")

        return results, ri_per_hour, zone_stats, figs, axes

    mcpserver.report = wrapped_report


_patch_mcp_report()


def load_all_patients() -> list[str]:
    patient_df = pd.read_csv(PATIENT_FILE)
    return patient_df["Name"].tolist()


def get_api_key(cli_value: str | None) -> str:
    if cli_value:
        return cli_value.strip()

    env_value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_value:
        return env_value

    return getpass("OpenRouter API key: ").strip()


def normalize_params(controller_type: str, params: dict) -> dict:
    if controller_type == "pid":
        return {
            "kp": round(float(params["kp"]), 8),
            "ki": round(float(params["ki"]), 8),
            "kd": round(float(params["kd"]), 8),
            "target_bg": round(float(params.get("target_bg", 110.0)), 2),
        }

    if controller_type == "bb":
        return {"cr_multiplier": round(float(params["cr_multiplier"]), 4)}

    raise ValueError(f"Unsupported controller_type '{controller_type}'.")


def params_key(controller_type: str, params: dict) -> str:
    return json.dumps(normalize_params(controller_type, params), sort_keys=True)


def seed_parameter_sets(controller_type: str) -> list[dict]:
    if controller_type == "pid":
        return [
            {"kp": 0.00025, "ki": 0.0, "kd": 0.0, "target_bg": 110.0},
            {"kp": 0.0005, "ki": 0.000025, "kd": 0.0, "target_bg": 110.0},
            {"kp": 0.00075, "ki": 0.00005, "kd": 0.0, "target_bg": 110.0},
            {"kp": 0.001, "ki": 0.00005, "kd": 0.0025, "target_bg": 110.0},
            {"kp": 0.00125, "ki": 0.0001, "kd": 0.0025, "target_bg": 105.0},
        ]

    if controller_type == "bb":
        return [
            {"cr_multiplier": 0.75},
            {"cr_multiplier": 0.9},
            {"cr_multiplier": 1.0},
            {"cr_multiplier": 1.1},
            {"cr_multiplier": 1.25},
        ]

    raise ValueError(f"Unsupported controller_type '{controller_type}'.")


def evaluate_candidate_via_mcp(
    patient_ids: list[str], controller_type: str, params: dict
) -> CandidateResult:
    normalized_params = normalize_params(controller_type, params)
    metric_rows = []

    for patient_id in patient_ids:
        result = mcpserver.tool_run_single_simulation(
            patient_id, controller_type, normalized_params
        )
        if result.get("status") != "ok":
            raise RuntimeError(
                f"Simulation failed for patient {patient_id}, controller {controller_type}, "
                f"params {normalized_params}: {result.get('message')}"
            )
        metrics = dict(result["metrics"])
        metrics["patient_id"] = patient_id
        metric_rows.append(metrics)

    metric_frame = pd.DataFrame(metric_rows)
    average_metrics = {
        column: round(float(metric_frame[column].mean()), 2)
        for column in metric_frame.columns
        if column != "patient_id"
    }
    worst_tir = round(float(metric_frame["TIR"].min()), 2)
    worst_hypo = round(float(metric_frame["Hypo"].max()), 2)

    return CandidateResult(
        controller_type=controller_type,
        params=normalized_params,
        average_metrics=average_metrics,
        worst_tir=worst_tir,
        worst_hypo=worst_hypo,
        patient_count=len(patient_ids),
    )


def rank_key(candidate: CandidateResult) -> tuple[float, float, float]:
    return (
        candidate.average_metrics["CompositeScore"],
        candidate.average_metrics["TIR"],
        -candidate.average_metrics["Hypo"],
    )


def candidate_results_to_frame(results: list[CandidateResult]) -> pd.DataFrame:
    rows = []
    for rank, result in enumerate(results, start=1):
        row = {
            "rank": rank,
            "controller_type": result.controller_type,
            "params_json": json.dumps(result.params, sort_keys=True),
            "worst_tir": result.worst_tir,
            "worst_hypo": result.worst_hypo,
            "patient_count": result.patient_count,
        }
        row.update({f"param_{key}": value for key, value in result.params.items()})
        row.update(result.average_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def save_candidate_rankings(results: list[CandidateResult], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = candidate_results_to_frame(results)
    frame.to_csv(output_dir / "rankings.csv", index=False)
    (output_dir / "rankings.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )
    return frame


def build_optimizer_prompts(
    controller_type: str,
    tested_count: int,
    budget: int,
    top_history: list[dict],
) -> tuple[str, str]:
    if controller_type == "pid":
        parameter_guide = (
            "PID parameters:\n"
            "- kp between 0.0001 and 0.002\n"
            "- ki between 0.0 and 0.00015\n"
            "- kd between 0.0 and 0.005\n"
            "- target_bg between 100 and 120\n"
        )
        json_shape = (
            '[{"kp": 0.0005, "ki": 0.00005, "kd": 0.0025, "target_bg": 110.0}]'
        )
    else:
        parameter_guide = "Basal-bolus parameter:\n- cr_multiplier between 0.6 and 1.6\n"
        json_shape = '[{"cr_multiplier": 1.0}]'

    system_prompt = (
        "You are optimizing diabetes controller parameters. "
        "Propose the next batch of parameter sets to test based on prior results. "
        "Maximize CompositeScore while protecting TIR and minimizing Hypo and CV. "
        "Return JSON only."
    )
    user_prompt = (
        f"Controller type: {controller_type}\n"
        f"Tested so far: {tested_count} / {budget}\n"
        f"Need next batch size: {min(PROPOSAL_BATCH_SIZE, budget - tested_count)}\n"
        f"{parameter_guide}\n"
        "Use the top history below to choose promising unexplored settings. "
        "Explore locally around strong candidates but keep some diversity.\n\n"
        f"Top history:\n{json.dumps(top_history, indent=2)}\n\n"
        "Return a JSON object with one key, candidates, whose value is a list of parameter objects.\n"
        f"Example shape:\n{{\"candidates\": {json_shape}}}"
    )
    return system_prompt, user_prompt


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    app_url: str | None,
    app_name: str | None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_name:
        headers["X-Title"] = app_name

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start : end + 1])


def propose_next_candidates(
    api_key: str,
    model: str,
    controller_type: str,
    tested_results: list[CandidateResult],
    tested_keys: set[str],
    app_url: str | None,
    app_name: str | None,
) -> list[dict]:
    top_history = candidate_results_to_frame(
        sorted(tested_results, key=rank_key, reverse=True)
    ).head(10).to_dict(orient="records")
    system_prompt, user_prompt = build_optimizer_prompts(
        controller_type, len(tested_results), OPTIMIZATION_BUDGET, top_history
    )
    raw_response = call_openrouter(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        app_url=app_url,
        app_name=app_name,
    )
    parsed = extract_json_object(raw_response)
    candidates = parsed.get("candidates", [])

    unique_candidates = []
    for candidate in candidates:
        normalized = normalize_params(controller_type, candidate)
        key = params_key(controller_type, normalized)
        if key in tested_keys:
            continue
        unique_candidates.append(normalized)
    return unique_candidates


def fallback_candidates(controller_type: str, tested_keys: set[str], count: int) -> list[dict]:
    if controller_type == "pid":
        kp_values = [0.0002, 0.00035, 0.0005, 0.00065, 0.0008, 0.001, 0.0012, 0.0015]
        ki_values = [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001]
        kd_values = [0.0, 0.001, 0.0025, 0.004]
        target_values = [100.0, 105.0, 110.0, 115.0, 120.0]
        pool = []
        for kp in kp_values:
            for ki in ki_values:
                for kd in kd_values:
                    for target_bg in target_values:
                        params = normalize_params(
                            "pid",
                            {"kp": kp, "ki": ki, "kd": kd, "target_bg": target_bg},
                        )
                        if params_key("pid", params) not in tested_keys:
                            pool.append(params)
        return pool[:count]

    pool = []
    for value in [round(0.6 + 0.01 * idx, 2) for idx in range(101)]:
        params = normalize_params("bb", {"cr_multiplier": value})
        if params_key("bb", params) not in tested_keys:
            pool.append(params)
    return pool[:count]


def optimize_controller_with_llm(
    patient_ids: list[str],
    controller_type: str,
    api_key: str,
    model: str,
    app_url: str | None,
    app_name: str | None,
) -> list[CandidateResult]:
    print(
        f"Optimizing {controller_type} across {len(patient_ids)} patients "
        f"with MCP-tool execution and LLM-guided search, budget {OPTIMIZATION_BUDGET}..."
    )
    tested_results: list[CandidateResult] = []
    tested_keys: set[str] = set()

    for params in seed_parameter_sets(controller_type):
        normalized = normalize_params(controller_type, params)
        print(f"  [{controller_type}] seed: {normalized}")
        result = evaluate_candidate_via_mcp(patient_ids, controller_type, normalized)
        tested_results.append(result)
        tested_keys.add(params_key(controller_type, normalized))

    while len(tested_results) < OPTIMIZATION_BUDGET:
        needed = min(PROPOSAL_BATCH_SIZE, OPTIMIZATION_BUDGET - len(tested_results))
        try:
            proposed = propose_next_candidates(
                api_key=api_key,
                model=model,
                controller_type=controller_type,
                tested_results=tested_results,
                tested_keys=tested_keys,
                app_url=app_url,
                app_name=app_name,
            )
        except Exception as exc:
            print(f"  [{controller_type}] LLM proposal failed: {exc}")
            proposed = []

        if not proposed:
            proposed = fallback_candidates(controller_type, tested_keys, needed)

        for params in proposed[:needed]:
            key = params_key(controller_type, params)
            if key in tested_keys:
                continue
            print(
                f"  [{controller_type}] llm set {len(tested_results) + 1}/{OPTIMIZATION_BUDGET}: "
                f"{params}"
            )
            result = evaluate_candidate_via_mcp(patient_ids, controller_type, params)
            tested_results.append(result)
            tested_keys.add(key)
            if len(tested_results) >= OPTIMIZATION_BUDGET:
                break

    return sorted(tested_results, key=rank_key, reverse=True)


def run_population_scenario_via_mcp(
    patient_ids: list[str],
    controller_type: str,
    params: dict,
    scenario: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running {controller_type} scenario '{scenario}' across {len(patient_ids)} patients via MCP tools..."
    )
    result = mcpserver.tool_validate_population_scenario(
        patient_ids=patient_ids,
        controller_type=controller_type,
        optimized_params=params,
        what_if_condition=scenario,
        save_path=str(output_dir),
    )
    if result.get("status") != "ok":
        raise RuntimeError(
            f"Population scenario validation failed for controller {controller_type}, "
            f"scenario {scenario}: {result.get('message')}"
        )

    patient_metrics_df = pd.read_csv(output_dir / "patient_metrics.csv")
    create_boxplot_summary(patient_metrics_df, controller_type, scenario, output_dir)
    return {
        "controller_type": result["controller_type"],
        "scenario": result["scenario"],
        "params": result["params"],
        "patient_count": result["patient_count"],
        "average_metrics": result["average_metrics"],
        "worst_tir": result["worst_tir"],
        "worst_hypo": result["worst_hypo"],
    }


def create_boxplot_summary(
    patient_metrics_df: pd.DataFrame,
    controller_type: str,
    scenario: str,
    output_dir: Path,
) -> None:
    tir_values = patient_metrics_df["TIR"].dropna().tolist()
    mean_tir = patient_metrics_df["TIR"].mean()
    rng = pd.Series(range(len(tir_values)))
    jitter = ((rng % 10) - 4.5) * 0.012

    fig, ax = plt.subplots(figsize=(7, 7))
    box = ax.boxplot(
        [tir_values],
        labels=["TIR"],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#d62828",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#f4a261",
            "markeredgecolor": "#7f5539",
            "markersize": 4,
            "alpha": 0.8,
        },
    )

    box["boxes"][0].set_facecolor("#2a9d8f")
    box["boxes"][0].set_alpha(0.65)

    ax.scatter(
        1 + jitter.to_numpy(),
        tir_values,
        s=34,
        alpha=0.75,
        color="#264653",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="Patients",
    )
    ax.axhline(mean_tir, color="#1d3557", linestyle="--", linewidth=1, label="Mean")
    ax.set_title(f"{controller_type.upper()} - {scenario} - TIR Distribution")
    ax.set_ylabel("TIR (%)")
    ax.set_xlim(0.7, 1.3)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "patient_boxplot.png", dpi=150)
    plt.close(fig)


def save_overall_summary(summaries: list[dict], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in summaries:
        row = {
            "controller_type": summary["controller_type"],
            "scenario": summary["scenario"],
            "patient_count": summary["patient_count"],
            "worst_tir": summary["worst_tir"],
            "worst_hypo": summary["worst_hypo"],
            "params_json": json.dumps(summary["params"], sort_keys=True),
        }
        row.update(summary["average_metrics"])
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["controller_type", "scenario"])
    frame.to_csv(output_dir / "overall_scenario_summary.csv", index=False)
    (output_dir / "overall_scenario_summary.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )
    return frame


def build_final_summary_prompts(
    pid_rankings: pd.DataFrame,
    bb_rankings: pd.DataFrame,
    scenario_summary: pd.DataFrame,
) -> tuple[str, str]:
    system_prompt = (
        "You are a careful diabetes simulation analyst. Compare PID and basal-bolus "
        "controller optimization results, explain tradeoffs briefly, and recommend "
        "which controller is stronger overall. Use the provided metrics only."
    )
    user_prompt = (
        "Summarize the optimization and scenario validation results.\n\n"
        "Requirements:\n"
        "- Identify the best PID parameters and best BB parameters.\n"
        "- Compare the two controllers on baseline optimization and on the validated scenarios.\n"
        "- Mention TIR, Hypo, CV, and CompositeScore.\n"
        "- State which controller is stronger overall and why.\n"
        "- Return a short section called recommended_params with JSON for both controllers.\n"
        "- Keep the answer concise and practical.\n\n"
        f"Top 10 PID results:\n{pid_rankings.head(10).to_json(orient='records', indent=2)}\n\n"
        f"Top 10 BB results:\n{bb_rankings.head(10).to_json(orient='records', indent=2)}\n\n"
        f"Scenario summary:\n{scenario_summary.to_json(orient='records', indent=2)}"
    )
    return system_prompt, user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use MCP tools for controller simulations, OpenRouter for optimization proposals, "
            "and generate population-level scenario summaries."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="agent_outputs",
        help="Directory where rankings, CSV/JSON files, and plots are saved.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=DEFAULT_SCENARIOS,
        help="Scenarios to evaluate after optimization.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5",
        help="OpenRouter model name for both optimization proposals and the final summary.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key. If omitted, the script uses OPENROUTER_API_KEY or prompts.",
    )
    parser.add_argument(
        "--app-url",
        default=None,
        help="Optional HTTP-Referer sent to OpenRouter.",
    )
    parser.add_argument(
        "--app-name",
        default="simglucose-agent",
        help="Optional X-Title sent to OpenRouter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = get_api_key(args.api_key)
    if not api_key:
        raise ValueError("An OpenRouter API key is required.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / args.output_dir / run_stamp
    optimization_dir = output_dir / "optimization"
    validation_dir = output_dir / "validation"
    llm_dir = output_dir / "llm"

    patient_ids = load_all_patients()
    print(f"Loaded {len(patient_ids)} virtual patients.")

    mcpserver.tool_reset_history()
    pid_results = optimize_controller_with_llm(
        patient_ids, "pid", api_key, args.model, args.app_url, args.app_name
    )
    pid_history = mcpserver.tool_get_history(controller_type="pid", top_n=10)

    mcpserver.tool_reset_history()
    bb_results = optimize_controller_with_llm(
        patient_ids, "bb", api_key, args.model, args.app_url, args.app_name
    )
    bb_history = mcpserver.tool_get_history(controller_type="bb", top_n=10)

    pid_ranking_frame = save_candidate_rankings(pid_results, optimization_dir / "pid")
    bb_ranking_frame = save_candidate_rankings(bb_results, optimization_dir / "bb")

    (optimization_dir / "pid" / "mcp_history.json").write_text(
        json.dumps(pid_history, indent=2), encoding="utf-8"
    )
    (optimization_dir / "bb" / "mcp_history.json").write_text(
        json.dumps(bb_history, indent=2), encoding="utf-8"
    )

    best_pid = pid_results[0]
    best_bb = bb_results[0]

    comparison_rows = candidate_results_to_frame([best_pid, best_bb])
    comparison_rows.to_csv(optimization_dir / "best_controller_comparison.csv", index=False)
    (optimization_dir / "best_controller_comparison.json").write_text(
        comparison_rows.to_json(orient="records", indent=2), encoding="utf-8"
    )

    scenario_summaries = []
    for controller_type, candidate in [("pid", best_pid), ("bb", best_bb)]:
        for scenario in args.scenarios:
            summary = run_population_scenario_via_mcp(
                patient_ids,
                controller_type,
                candidate.params,
                scenario,
                validation_dir / controller_type / scenario,
            )
            scenario_summaries.append(summary)

    scenario_summary_frame = save_overall_summary(scenario_summaries, validation_dir)

    system_prompt, user_prompt = build_final_summary_prompts(
        pid_ranking_frame, bb_ranking_frame, scenario_summary_frame
    )
    print(f"Requesting final summary from OpenRouter model '{args.model}'...")
    llm_summary = call_openrouter(
        api_key=api_key,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        app_url=args.app_url,
        app_name=args.app_name,
    )

    llm_dir.mkdir(parents=True, exist_ok=True)
    (llm_dir / "final_summary.md").write_text(llm_summary, encoding="utf-8")
    (llm_dir / "final_summary_prompt.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nOptimization complete.")
    print(f"Outputs saved under: {output_dir}")
    print(f"Best PID params: {best_pid.params}")
    print(f"Best BB params: {best_bb.params}")
    print(f"LLM summary saved to: {llm_dir / 'final_summary.md'}")


if __name__ == "__main__":
    main()
