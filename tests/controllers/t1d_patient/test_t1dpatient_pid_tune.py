import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from enum import Enum

from test_t1dpatient_pid import (
    run_sim_simple_pid_no_meal,
    run_sim_simple_pid_single_meal,
    run_sim_simple_pid_attack,
)
from utils import get_patients
from simglucose.simulation.scenario_simple import Scenario

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent
result_folder = parent_folder / "results"


def test_range_simple_pid_no_meal(
    k_p: list,
    k_i: list,
    k_d: list,
    sample_time: list,
    basal_rate: list,
    csv_name,
    save_csv=False,
):
    from itertools import product

    csv_file = f"{csv_name}.csv"
    combinations = list(product(k_p, k_i, k_d, sample_time, basal_rate))
    results = []
    for combination in combinations:
        kp, ki, kd, st, br = combination
        rmse = run_sim_simple_pid_no_meal(kp, ki, kd, st, br)
        results.append(rmse)

    if save_csv:
        with open(csv_file, "w") as f:
            f.write(f"k_p,k_i,k_d,sample_time,basal_rate,rmse\n")
            for i in range(len(combinations)):
                f.write(
                    f"{combinations[i][0]},{combinations[i][1]},{combinations[i][2]},{combinations[i][3]},{combinations[i][4]},{results[i]}\n"
                )


def run_single_simulation(
    params: tuple,
    sample_time: int,
    sim_time: int,
    scenario: Scenario,
):
    kp, ki, kd, br, patient_name = params
    try:
        if scenario == Scenario.NO_MEAL:
            rmse = run_sim_simple_pid_no_meal(
                k_P=kp,
                k_I=ki,
                k_D=kd,
                sample_time=sample_time,
                basal_rate=br,
                patient_name=patient_name,
                sim_time=sim_time,
                save_fig=False,
                show_fig=False,
                log=False,
            )
            return {
                "k_p": kp,
                "k_i": ki,
                "k_d": kd,
                "basal_rate": br,
                "patient_name": patient_name,
                "rmse": rmse,
            }
        elif scenario == Scenario.SINGLE_MEAL:
            rmse = run_sim_simple_pid_single_meal(
                k_P=kp,
                k_I=ki,
                k_D=kd,
                sample_time=sample_time,
                basal_rate=br,
                patient_name=patient_name,
                sim_time=sim_time,
                save_fig=False,
                show_fig=False,
                log=False,
            )
            return {
                "k_p": kp,
                "k_i": ki,
                "k_d": kd,
                "basal_rate": br,
                "patient_name": patient_name,
                "rmse": rmse,
            }
        # elif scenario == Scenario.ATTACK: # skip attack for now
        else:
            raise ValueError(f"Invalid scenario: {scenario}")
    except Exception as e:
        print(f"Error running simulation with params {params}: {str(e)}")
        return None


def parallel_test_pid_parameters(
    k_p_range=[1e-6, 1e-5, 1e-4, 1e-3],
    k_i_range=[0, 1e-10, 1e-9, 1e-8, 1e-7],
    k_d_range=[0, 1e-8, 1e-7],
    sample_time=5,
    basal_rate=[0, 0.05, 0.1, 0.15, 0.2],
    sim_time=2000,
    n_jobs=4,
    scenario: Scenario = Scenario.NO_MEAL,
):
    """
    Run parallel simulations to test different PID parameters
    """
    from multiprocessing import Pool
    from itertools import product
    from functools import partial
    from tqdm import tqdm

    # Generate parameter combinations
    patients = get_patients()

    param_combinations = list(
        product(k_p_range, k_i_range, k_d_range, basal_rate, patients)
    )

    # Filter combinations where ki is smaller than kp (if ki is not 0)
    valid_combinations = [
        (kp, ki, kd, br, patient_name)
        for kp, ki, kd, br, patient_name in param_combinations
        if ki == 0 or ki < kp
    ]

    print(f"Testing {len(valid_combinations)} parameter combinations...")

    simulation_with_fixed_params = partial(
        run_single_simulation,
        sample_time=sample_time,
        sim_time=sim_time,
        scenario=scenario,
    )

    # Run parallel simulations
    with Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(simulation_with_fixed_params, valid_combinations),
                total=len(valid_combinations),
                desc="Running simulations",
                unit="sim",
                ncols=100,  # Fixed width for cleaner output
            )
        )

    # Filter out failed simulations and convert to DataFrame
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    # Find best 5 parameters for each patient and dump to one json file
    best_params_dict = {}
    for patient_name in patients:
        patient_results = df[df["patient_name"] == patient_name]
        if not patient_results.empty:
            best_params = patient_results.sort_values(by="rmse").head(5)
            best_params_dict[patient_name] = best_params.to_dict(orient="records")
            print(f"\nBest parameters for {patient_name}:")
            for i, row in best_params.iterrows():
                print(
                    f"k_p={row['k_p']}, k_i={row['k_i']}, k_d={row['k_d']}, "
                    f"basal_rate={row['basal_rate']}, rmse={row['rmse']:.4f}"
                )

    json_file = result_folder / (
        f"pid_{scenario.value}_tunning_step5_{sample_time}min_{sim_time}min.json"
    )
    with open(json_file, "w") as f:
        json.dump(best_params_dict, f)

    return json_file


def plot_best_N_params(
    json_file: str,
    sample_time: int,
    sim_time: int,
    N: int = 1,
    save_fig=True,
):
    if N > 5:
        N = 5

    with open(json_file, "r") as f:
        best_params_by_patient = json.load(f)

    scenario = Path(json_file).stem.split("_")[1]
    if scenario == "no":
        scenario = Scenario.NO_MEAL
    elif scenario == "single":
        scenario = Scenario.SINGLE_MEAL
    else:
        raise ValueError(f"Invalid scenario: {scenario}")

    for patient_name, best_params in best_params_by_patient.items():
        print(f"patient_name: {patient_name}")
        for i, row in enumerate(best_params):
            if i + 1 > N:
                break
            print(
                f"No.{i+1}: "
                f"k_p={row['k_p']}, k_i={row['k_i']}, k_d={row['k_d']}, "
                f"basal_rate={row['basal_rate']}, rmse={row['rmse']:.4f}"
            )

            if scenario == Scenario.NO_MEAL:
                # Run simulation with best parameters and save plot
                run_sim_simple_pid_no_meal(
                    k_P=row["k_p"],
                    k_I=row["k_i"],
                    k_D=row["k_d"],
                    sample_time=sample_time,
                    basal_rate=row["basal_rate"],
                    patient_name=patient_name,
                    sim_time=sim_time,
                    save_fig=save_fig,
                    show_fig=False,
                    log=False,
                )
            elif scenario == Scenario.SINGLE_MEAL:
                run_sim_simple_pid_single_meal(
                    k_P=row["k_p"],
                    k_I=row["k_i"],
                    k_D=row["k_d"],
                    sample_time=sample_time,
                    basal_rate=row["basal_rate"],
                    patient_name=patient_name,
                    sim_time=sim_time,
                    save_fig=save_fig,
                    show_fig=False,
                    log=False,
                )


def find_best_params_by_group(
    source_json_file_name: str,
    save_json_file_name: str,
):
    # find average params for each patient group
    groups = ["adolescent", "adult", "child"]
    groups_params = {"adolescent": {}, "adult": {}, "child": {}}
    source_json_file = result_folder / source_json_file_name

    with open(source_json_file, "r") as f:
        best_params_by_patient = json.load(f)

    # average for first N params
    N = 1
    for patient_name, best_params in best_params_by_patient.items():
        group = patient_name.split("#")[0]
        print(f"patient_name: {patient_name}, group: {group}", end=" ")
        for p in best_params[:N]:
            groups_params[group].setdefault("k_p", []).append(p["k_p"])
            groups_params[group].setdefault("k_i", []).append(p["k_i"])
            groups_params[group].setdefault("k_d", []).append(p["k_d"])
            groups_params[group].setdefault("basal_rate", []).append(p["basal_rate"])
            print(
                f"k_p: {p['k_p']}, k_i: {p['k_i']}, k_d: {p['k_d']}, basal_rate: {p['basal_rate']}"
            )

    save_json_file = result_folder / save_json_file_name

    with open(save_json_file, "w") as f:
        json.dump(groups_params, f)

    for group, params in groups_params.items():
        print(f"group: {group}")
        for k, v in params.items():
            print(f"{k}: {np.mean(v):.1e}")


if __name__ == "__main__":
    # test code
    # json_file = json_file = parallel_test_pid_parameters(
    #     k_p_range=[1e-9],
    #     k_i_range=[0],
    #     k_d_range=[0],
    #     basal_rate=[0.1],
    #     n_jobs=8,
    #     scenario=Scenario.SINGLE_MEAL,
    # )

    # run in hyades
    # json_file = parallel_test_pid_parameters(
    #     k_p_range=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #     k_i_range=[0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #     k_d_range=[0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #     basal_rate=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    #     n_jobs=64,
    #     scenario=Scenario.NO_MEAL,
    # )

    # json_file = parallel_test_pid_parameters(
    #     k_p_range=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #     k_i_range=[0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #     k_d_range=[0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #     basal_rate=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    #     n_jobs=64,
    #     scenario=Scenario.SINGLE_MEAL,
    # )

    # json_file = parallel_test_pid_parameters(
    #     k_p_range=[1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1],
    #     k_i_range=[0, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    #     k_d_range=[0, 1e-2, 5e-2, 0.1, 0.3, 0.5, 0.7, 1],
    #     basal_rate=[0, 0.04, 0.08, 0.1, 0.2],
    #     n_jobs=64,
    #     scenario=Scenario.SINGLE_MEAL,
    # )

    # plot best 5 params
    # json_file = (
    #     result_folder / "pid_single_meal_tunning_step5_5min_2000min_refined.json"
    # )
    # plot_best_N_params(
    #     json_file=json_file,
    #     sample_time=5,
    #     sim_time=2000,
    #     N=1,
    # )

    # find best params by group

    # find_best_params_by_group(
    #     source_json_file_name="pid_no_meal_tunning_step5_5min_2000min_refined.json",
    #     save_json_file_name="pid_no_meal_tunning_step5_5min_2000min_refined_avg_first1.json",
    # )

    # find_best_params_by_group(
    #     source_json_file_name="pid_no_meal_tunning_step5_5min_2000min.json",
    #     save_json_file_name="pid_no_meal_tunning_step5_5min_2000min_avg_first1.json",
    # )

    # find_best_params_by_group(
    #     source_json_file_name="pid_single_meal_tunning_step5_5min_2000min.json",
    #     save_json_file_name="pid_single_meal_tunning_step5_5min_2000min_avg_first1.json",
    # )

    # find_best_params_by_group(
    #     source_json_file_name="pid_single_meal_tunning_step5_5min_2000min_refined.json",
    #     save_json_file_name="pid_single_meal_tunning_step5_5min_2000min_refined_avg_first1.json",
    # )

    pass
