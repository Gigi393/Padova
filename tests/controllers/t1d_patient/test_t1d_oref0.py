"""
Simple test script for ORef0 controller with T1D patients.
Based on the simulation pattern in t1dpatient_2.py
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from simglucose.patient.t1dpatient_2 import T1DPatient, Action
from simglucose.controller.oref_zero import ORefZeroController, CtrlObservation
from simglucose.simulation.scenario_simple import Scenario
from simglucose.utils import fetch_patient_quest
from analytics import PatientType
from plotting import plot_and_save
from utils import get_patient_by_group

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"


def test_oref0_simulation(
    patient_name="adolescent#003",
    scenario=Scenario.SINGLE_MEAL,
    save_fig=True,
):
    """
    Run a simple ORef0 simulation with T1D patient.

    Args:
        patient_name: Name of the patient
        scenario: Meal scenario from test_utils
        simulation_time: Total simulation time in minutes
        save_plot: Whether to save the plot
    """
    # Initialize patient
    p = T1DPatient.withName(patient_name)
    profile = {}

    params = fetch_patient_quest(patient_name)
    profile["carb_ratio"] = params["CR"]
    profile["current_basal"] = p._params.u2ss * p._params.BW / 6000 * 60  # to U/h
    # Initialize controller with minimal configuration
    ctrl = ORefZeroController(patient_name=patient_name, profile=profile)
    if not ctrl.initialize():
        raise ValueError("Failed to initialize Oref0 controller")
    logger.info(f"Patient {patient_name} and ORef0 controller initialized")
    # Data collection lists
    t = []
    CHO = []
    insulin = []
    BG = []

    test_patient_dir = img_dir / f"test_oref0"
    test_patient_dir.mkdir(exist_ok=True)

    # Simulation start time
    current_sim_time = datetime.now()

    print(f"Starting simulation for {patient_name} with {scenario.name} scenario")
    print(
        f"Simulation duration: {scenario.max_t} minutes ({scenario.max_t/60:.1f} hours)"
    )

    # Main simulation loop
    while p.t < scenario.max_t:
        # Get carb intake based on scenario and exact time
        carb = scenario.get_carb(int(p.t), p._params.BW)

        # Create controller observation
        ctrl_obs = CtrlObservation(p.observation.Gsub, bolus=0)

        # Check for critical hypoglycemia
        if p.observation.Gsub < 39:
            print(f"Critical hypoglycemia at t={p.t}, BG={p.observation.Gsub}")
            break

        # Get controller action
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            meal=carb,
            time=current_sim_time,
        )

        # Calculate total insulin
        ins = ctrl_action.basal + ctrl_action.bolus

        # Create patient action
        act = Action(insulin=ins, CHO=carb)

        # Record data
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)

        # Step patient forward
        p.step(act)

        # Update simulation time
        current_sim_time += timedelta(minutes=p.sample_time)
        print(
            f"\033[94mt: {p.t}, time: {current_sim_time} BG: {p.observation.Gsub}, CHO: {carb}, Insulin: {ins}\033[0m"
        )

    # Calculate basic statistics
    import numpy as np

    BG_array = np.array(BG)
    time_in_range = np.sum((BG_array >= 70) & (BG_array <= 180)) / len(BG_array) * 100
    mean_bg = np.mean(BG_array)
    std_bg = np.std(BG_array)

    print(f"\nSimulation Results:")
    print(f"  Time in Range (70-180): {time_in_range:.1f}%")
    print(f"  Mean BG: {mean_bg:.1f} mg/dL")
    print(f"  Std BG: {std_bg:.1f} mg/dL")
    print(f"  Min BG: {np.min(BG_array):.1f} mg/dL")
    print(f"  Max BG: {np.max(BG_array):.1f} mg/dL")

    # Create plot using plot_and_save function from test_utils
    if save_fig:
        filename = test_patient_dir / f"oref0_{patient_name}_{scenario.name}.png"
        plot_and_save(
            t,
            BG,
            CHO,
            insulin,
            target_BG=ctrl.target_bg,
            file_name=str(filename),
        )

    return {
        "time": t,
        "BG": BG,
        "CHO": CHO,
        "insulin": insulin,
        "time_in_range": time_in_range,
        "mean_bg": mean_bg,
        "std_bg": std_bg,
    }


if __name__ == "__main__":
    # Test single meal scenario

    patient_groups = [
        PatientType.ADOLESCENT,
        PatientType.ADULT,
        PatientType.CHILD,
    ]
    patients = []
    for group in patient_groups:
        group_patients = get_patient_by_group(group)
        if group_patients:
            patients.extend(group_patients)

    for patient_name in patients:
        test_oref0_simulation(
            patient_name=patient_name,
            save_fig=True,
            scenario=Scenario.SINGLE_MEAL,
        )
