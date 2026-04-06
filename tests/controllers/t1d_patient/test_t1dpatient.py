import logging
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path


from simglucose.patient.t1dpatient import T1DPatient, Action
from plotting import plot_and_show, plot_and_save
from analytics import get_rmse

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
best_basal_folder = parent_folder / "results" / "best_basal_rate"


def test_patient(
    patient_name="adolescent#003",
    basal_rate=None,
    fig_title=None,
    save_fig=False,
    show_fig=False,
):
    test_patient_dir = img_dir / f"test_patient"
    test_patient_dir.mkdir(exist_ok=True)

    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    while p.t < 2000:
        carb = 0

        if basal_rate is not None:
            act = Action(insulin=basal_rate, CHO=carb)
        else:
            act = Action(insulin=0, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    if fig_title is None:
        if basal_rate is not None:
            fig_title = f"test_patient_{patient_name}_basal_{basal_rate}"
        else:
            fig_title = f"test_patient_{patient_name}_no_basal"

    if show_fig:
        plot_and_show(t, BG, CHO, insulin, BG[0], fig_title)
    if save_fig:
        file_name = test_patient_dir / f"{fig_title}.png"
        plot_and_save(t, BG, CHO, insulin, BG[0], file_name)


def patient_no_meal_rmse(patient_name="adolescent#003", basal_rate=0.2, sim_time=1000):
    p = T1DPatient.withName(patient_name)
    t = []
    CHO = []
    insulin = []
    BG = []

    target_BG = None
    while p.t < sim_time:
        carb = 0
        act = Action(insulin=basal_rate, CHO=carb)
        if target_BG is None:
            target_BG = p.observation.Gsub
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    rmse = get_rmse(BG, target_BG)  # stable => rmse is small
    return rmse


def run_single_simulation(
    params: tuple,
    sim_time: int,
):
    patient_name, basal_rate = params
    try:
        rmse = patient_no_meal_rmse(patient_name, basal_rate, sim_time)
        return {
            "patient_name": patient_name,
            "basal_rate": basal_rate,
            "rmse": rmse,
        }
    except Exception as e:
        print(f"Error running simulation with params {params}: {str(e)}")
        return None


def parallel_test_patient_no_meal_rmse(
    patient_name: list,
    basal_rate: list,
    sim_time: int,
    n_jobs: int,
    name_remark: str = "",
):
    from multiprocessing import Pool
    from itertools import product
    from functools import partial

    param_combinations = list(product(patient_name, basal_rate))

    simulation_with_fixed_params = partial(
        run_single_simulation,
        sim_time=sim_time,
    )
    with Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(simulation_with_fixed_params, param_combinations),
                total=len(param_combinations),
                desc="Running simulations",
                unit="sim",
                ncols=100,
            )
        )

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    # find the best basal rate for each patient
    best_basal_rate = {}
    for patient_name in patient_name:
        patient_results = df[df["patient_name"] == patient_name]
        if not patient_results.empty:
            best_param = patient_results.sort_values(by="rmse").head(1)
            best_basal_rate[patient_name] = best_param["basal_rate"].values[0]
            print(
                f"Best basal rate for {patient_name}: {best_basal_rate[patient_name]}"
            )

    # dump to json file
    best_basal_folder.mkdir(exist_ok=True, parents=True)
    json_file = best_basal_folder / f"{name_remark}patient_best_basal_rate.json"
    with open(json_file, "w") as f:
        json.dump(best_basal_rate, f)
    return json_file


def get_best_basal_rate():
    # get adolescent best basal rate
    patient_dict = {
        "adolescent": {
            "patient_name": [f"adolescent#00{i}" for i in range(1, 10)],
            "basal_rate": [
                0.009,
                0.01,
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
            ],
            "name_remark": "adolescent_",
        },
        "adult": {
            "patient_name": [f"adult#00{i}" for i in range(1, 10)],
            "basal_rate": [
                0.015,
                0.016,
                0.017,
                0.018,
                0.019,
                0.02,
                0.021,
                0.022,
                0.023,
                0.024,
            ],
            "name_remark": "adult_",
        },
        "child": {
            "patient_name": [f"child#00{i}" for i in range(1, 10)],
            "basal_rate": [
                0.005,
                0.0055,
                0.006,
                0.0065,
                0.007,
                0.0075,
                0.008,
            ],
            "name_remark": "child_",
        },
    }

    for _, patient_info in patient_dict.items():
        patient_name = patient_info["patient_name"]
        basal_rate = patient_info["basal_rate"]
        name_remark = patient_info["name_remark"]
        parallel_test_patient_no_meal_rmse(
            patient_name=patient_name,
            basal_rate=basal_rate,
            sim_time=1000,
            n_jobs=8,
            name_remark=name_remark,
        )


def combine_basal_rate():
    # get adolescent best basal rate
    adolescent_best_basal_rate = json.load(
        open(best_basal_folder / "adolescent_patient_best_basal_rate.json")
    )
    adult_best_basal_rate = json.load(
        open(best_basal_folder / "adult_patient_best_basal_rate.json")
    )
    child_best_basal_rate = json.load(
        open(best_basal_folder / "child_patient_best_basal_rate.json")
    )
    patient_best_basal_rate = {
        **adolescent_best_basal_rate,
        **adult_best_basal_rate,
        **child_best_basal_rate,
    }
    with open(best_basal_folder / "patient_best_basal_rate.json", "w") as f:
        json.dump(patient_best_basal_rate, f)


def save_best_basal_rate_fig():
    patient_best_basal_rate = json.load(
        open(best_basal_folder / "patient_best_basal_rate.json")
    )

    for patient_name, basal_rate in patient_best_basal_rate.items():
        test_patient(patient_name=patient_name, basal_rate=basal_rate, save_fig=True)


if __name__ == "__main__":
    # test patient and verify the best basal rate
    # test_patient(patient_name="adolescent#003", use_basal=True, basal_rate=0.011)
    # test_patient(patient_name="adolescent#001", use_basal=True, basal_rate=0.014)
    # test_patient(patient_name="adolescent#006", use_basal=True, basal_rate=0.015)
    # test_patient(patient_name="adolescent#008", use_basal=True, basal_rate=0.009)

    # test_patient(patient_name="adult#001", use_basal=True, basal_rate=0.021)
    # test_patient(patient_name="adult#005", use_basal=True, basal_rate=0.020)
    # test_patient(patient_name="adult#002", use_basal=True, basal_rate=0.022)
    # test_patient(patient_name="adult#003", use_basal=True, basal_rate=0.024)
    # test_patient(patient_name="adult#004", use_basal=True, basal_rate=0.015)

    # test_patient(patient_name="child#001", use_basal=True, basal_rate=0.0065)
    # test_patient(patient_name="child#002", use_basal=True, basal_rate=0.0065)
    # test_patient(patient_name="child#003", use_basal=True, basal_rate=0.005)
    # test_patient(patient_name="child#004", use_basal=True, basal_rate=0.008)
    # test_patient(patient_name="child#005", use_basal=True, basal_rate=0.008)
    # test_patient(patient_name="child#007", use_basal=True, basal_rate=0.008)

    # get best basal rate
    # get_best_basal_rate()

    # combine basal rate
    # combine_basal_rate()

    # save best basal rate fig
    save_best_basal_rate_fig()
