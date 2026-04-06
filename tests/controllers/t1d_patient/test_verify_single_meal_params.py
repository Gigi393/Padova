import json
import numpy as np
from enum import Enum
from pathlib import Path
from test_t1dpatient_pid import (
    run_sim_simple_pid_no_meal,
    run_sim_simple_pid_single_meal,
)
from analytics import PatientType
from simglucose.simulation.scenario_simple import Scenario
from utils import get_patient_by_group as _get_patient_by_group


class GoodParamCriteria(Enum):
    MOST_COMMON = "most_common"
    MEAN = "mean"


def get_good_param(lists: list, criteria: GoodParamCriteria):
    if criteria == GoodParamCriteria.MOST_COMMON:
        values, counts = np.unique(lists, return_counts=True)
        most_common_value = values[np.argmax(counts)]
        return most_common_value
    elif criteria == GoodParamCriteria.MEAN:
        return np.mean(lists)


def get_patient_by_group(patient_type: str):
    """Wrapper that converts string patient type to PatientType enum."""
    # Convert string to PatientType enum
    patient_type_enum = PatientType(patient_type)
    return _get_patient_by_group(patient_type_enum)


def run_single_meal_params_in_no_meal():
    # run single meal params in no meal scenario
    result_folder = Path(__file__).parent / "results"
    json_file = (
        result_folder / "pid_single_meal_tunning_step5_5min_2000min_refined.json"
    )

    with open(json_file, "r") as f:
        data = json.load(f)
    for patient, params in data.items():
        for ps in params:
            k_p = ps["k_p"]
            k_i = ps["k_i"]
            k_d = ps["k_d"]
            basal_rate = ps["basal_rate"]
            patient_name = ps["patient_name"]
            run_sim_simple_pid_no_meal(
                k_P=k_p,
                k_I=k_i,
                k_D=k_d,
                sample_time=5,
                basal_rate=basal_rate,
                patient_name=patient_name,
                sim_time=2000,
                save_fig=True,
                show_fig=False,
                log=False,
                img_folder_name=f"no_meal_with_single_meal_params",
            )
            break


def run_same_single_meal_params_in_scenario(
    criteria: GoodParamCriteria, scenario: Scenario
):
    # run single meal params in no meal scenario
    result_folder = Path(__file__).parent / "results"
    json_file = (
        result_folder
        / "pid_single_meal_tunning_step5_5min_2000min_refined_avg_first1.json"
    )
    with open(json_file, "r") as f:
        data = json.load(f)
    for patient_type, params in data.items():
        k_ps = np.array(params["k_p"])
        k_is = np.array(params["k_i"])
        k_ds = np.array(params["k_d"])
        basal_rates = np.array(params["basal_rate"])
        good_kp = get_good_param(k_ps, criteria)
        good_ki = get_good_param(k_is, criteria)
        good_kd = get_good_param(k_ds, criteria)
        good_br = get_good_param(basal_rates, criteria)
        print(f"{patient_type}: {good_kp}, {good_ki}, {good_kd}, {good_br}")
        patients = get_patient_by_group(patient_type)
        for patient_name in patients:
            if scenario == Scenario.SINGLE_MEAL:
                run_sim_simple_pid_single_meal(
                    k_P=good_kp,
                    k_I=good_ki,
                    k_D=good_kd,
                    sample_time=5,
                    basal_rate=good_br,
                    patient_name=patient_name,
                    sim_time=2000,
                    save_fig=True,
                    show_fig=False,
                    log=False,
                    img_folder_name=f"{scenario.value}_with_single_meal_params_{criteria.value}",
                )
            elif scenario == Scenario.NO_MEAL:
                run_sim_simple_pid_no_meal(
                    k_P=good_kp,
                    k_I=good_ki,
                    k_D=good_kd,
                    sample_time=5,
                    basal_rate=good_br,
                    patient_name=patient_name,
                    sim_time=2000,
                    save_fig=True,
                    show_fig=False,
                    log=False,
                    img_folder_name=f"{scenario.value}_with_single_meal_params_{criteria.value}",
                )


if __name__ == "__main__":
    # run_same_single_meal_params_in_scenario(
    #     GoodParamCriteria.MOST_COMMON, Scenario.SINGLE_MEAL
    # )
    # run_same_single_meal_params_in_scenario(
    #     GoodParamCriteria.MEAN, Scenario.SINGLE_MEAL
    # )
    run_single_meal_params_in_no_meal()
