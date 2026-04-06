"""
Simple example demonstrating how to use T1DM patient with ORef0 controller.

This script shows a basic simulation of a T1DM patient using the OpenAPS ORef0
algorithm for automated insulin delivery. It's meant as a reference implementation
and testing example.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero import ORefZeroController, CtrlObservation
from simglucose.simulation.scenario_simple import Scenario
from analytics import TIRConfig, PatientType
from plotting import plot_and_show_with_tir

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def plot_iob_comparison(t, patient_model_iob, openaps_iob, title="IOB Comparison"):
    """
    Plot IOB comparison between patient model and OpenAPS algorithm.

    Args:
        t: Time array in minutes
        patient_model_iob: List of patient model IOB values (U)
        openaps_iob: List of OpenAPS IOB values (U)
        title: Plot title
    """
    # Convert time to hours for better readability
    t_hours = np.array(t) / 60.0

    # Calculate difference
    iob_diff = np.array(patient_model_iob) - np.array(openaps_iob)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Both IOB values
    ax1.plot(
        t_hours,
        patient_model_iob,
        "b-",
        linewidth=2,
        label="Patient Model IOB",
        alpha=0.8,
    )
    ax1.plot(t_hours, openaps_iob, "r--", linewidth=2, label="OpenAPS IOB", alpha=0.8)
    ax1.set_ylabel("IOB (U)", fontsize=12, fontweight="bold")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    # Subplot 2: Difference
    ax2.plot(
        t_hours,
        iob_diff,
        "g-",
        linewidth=2,
        label="Difference (Model - OpenAPS)",
        alpha=0.8,
    )
    ax2.fill_between(t_hours, 0, iob_diff, alpha=0.3, color="g")
    ax2.set_xlabel("Time (hours)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("IOB Difference (U)", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    # Add statistics text box to difference plot
    stats_text = (
        f"Mean: {np.mean(iob_diff):.3f} U\n"
        f"Std Dev: {np.std(iob_diff):.3f} U\n"
        f"Max |Diff|: {np.max(np.abs(iob_diff)):.3f} U"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


def run_patient_with_oref0(
    patient_name="adult#001",
    scenario=Scenario.ONE_DAY,
    profile=None,
    show_plot=True,
):
    """
    Run a simulation of a T1DM patient with ORef0 controller.

    Args:
        patient_name: Name of the patient (e.g., "adult#001", "child#001")
        scenario: Meal scenario to simulate (NO_MEAL, SINGLE_MEAL, ONE_DAY, THREE_DAY)
        profile: Optional ORef0 profile dict with parameters like sens, dia, carb_ratio, etc.
        show_plot: Whether to display the results plot

    Returns:
        dict: Time in range statistics
    """
    # Initialize patient
    p = T1DMPatient.withName(patient_name)
    logger.info(f"Patient {patient_name} initialized")
    if profile is not None:
        profile["carb_ratio"] = p.carb_ratio
        profile["current_basal"] = p.basal * 60  # U/min to U/h

    # Initialize controller
    ctrl = ORefZeroController(patient_name=patient_name, profile=profile)
    if not ctrl.initialize():
        raise ValueError("Failed to initialize Oref0 controller")

    logger.info("ORef0 controller initialized")

    # Storage for simulation data
    t = []
    CHO = []
    insulin = []
    BG = []
    patient_model_iob = []  # IOB from patient physiological model
    openaps_iob = []  # IOB calculated by OpenAPS algorithm

    # Run simulation
    logger.info(f"Starting simulation with scenario: {scenario.name}")
    while p.t_elapsed < scenario.max_t:
        # Get meal for current time
        carb = scenario.get_carb(p.t_elapsed, p.body_weight)

        # Create observation for controller
        ctrl_obs = CtrlObservation(p.observation.Gsub, bolus=0)

        # Check for severe hypoglycemia
        if p.observation.Gsub < 39:
            logger.error("Severe hypoglycemia detected - stopping simulation")
            break

        # Get controller action
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            meal=carb,
            time=p.t,
        )

        # Calculate total insulin (basal + bolus)
        ins = ctrl_action.basal + ctrl_action.bolus
        act = Action(insulin=ins, CHO=carb)

        # Record data
        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)

        # Get IOB from patient model (physiological)
        # Use subtract_baseline=True to make it comparable with OpenAPS IOB
        # (both will measure insulin above baseline basal)
        model_iob = p.get_iob(include_plasma=True, subtract_baseline=False)
        patient_model_iob.append(model_iob if model_iob is not None else 0.0)

        # Get IOB from OpenAPS controller
        oaps_iob_data = ctrl.get_iob()
        oaps_iob_value = oaps_iob_data["iob_value"] if oaps_iob_data else 0.0
        openaps_iob.append(oaps_iob_value)

        # Step the patient simulation
        p.step(act)

        # Log progress (every hour)
        if p.t_elapsed % 60 == 0:
            logger.info(
                f"Time: {p.t_elapsed}min, BG: {p.observation.Gsub:.1f} mg/dL, "
                f"CHO: {carb}g, Insulin: {ins:.3f} U/min"
            )

    # Calculate time in range statistics using BASIC standard
    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(BG)

    logger.info("\n=== Time in Range Results ===")
    for category, percentage in time_in_range.items():
        logger.info(f"{category.value}: {percentage:.1f}%")

    # Display plot if requested
    if show_plot:
        plot_and_show_with_tir(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_bg,
            f"T1DM Patient {patient_name} with ORef0 - {scenario.name}",
            time_in_range,
            tir_config,
        )

        # Plot IOB comparison
        plot_iob_comparison(
            t,
            patient_model_iob,
            openaps_iob,
            title=f"IOB Comparison: {patient_name} - {scenario.name}",
        )

    # Log IOB comparison statistics
    logger.info("\n=== IOB Comparison (Insulin Above Baseline) ===")
    logger.info(
        f"Patient Model IOB - Mean: {np.mean(patient_model_iob):.2f}U, Max: {np.max(patient_model_iob):.2f}U, Min: {np.min(patient_model_iob):.2f}U"
    )
    logger.info(
        f"OpenAPS IOB      - Mean: {np.mean(openaps_iob):.2f}U, Max: {np.max(openaps_iob):.2f}U, Min: {np.min(openaps_iob):.2f}U"
    )

    # Calculate correlation and difference
    iob_diff = [pm - oa for pm, oa in zip(patient_model_iob, openaps_iob)]
    logger.info(f"\nDifference Statistics:")
    logger.info(f"  Mean Difference (Model - OpenAPS): {np.mean(iob_diff):.2f}U")
    logger.info(f"  Max Absolute Difference: {np.max(np.abs(iob_diff)):.2f}U")
    logger.info(f"  Std Dev of Difference: {np.std(iob_diff):.2f}U")
    logger.info(f"\nNote: Both IOB values measure insulin above baseline basal rate.")

    return {
        "time_in_range": time_in_range,
        "patient_model_iob": patient_model_iob,
        "openaps_iob": openaps_iob,
        "time": t,
    }


if __name__ == "__main__":
    # Example: Run with custom profile
    # refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    print("\n" + "=" * 60)
    print("Example: Adult patient with custom ORef0 profile")
    print("=" * 60)

    custom_profile = {
        "sens": 45,
        "dia": 7.0,
        "carb_ratio": 10,
        "max_iob": 20,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 5,  # from paper, max 10
        "max_daily_basal": 0.9,  # from paper
        "max_bg": 140,
        "min_bg": 90,
        "maxCOB": 120,  # from oref0 code
        "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 45}]},
        "min_5m_carbimpact": 8,  # from paper and oref0 code
    }

    result = run_patient_with_oref0(
        patient_name="adult#007",
        scenario=Scenario.SINGLE_MEAL,
        profile=custom_profile,
        show_plot=True,
    )

    # Check if time in range is acceptable using BASIC standard
    tir_config = TIRConfig()  # Defaults to BASIC standard
    results, count = tir_config.get_time_in_range_acceptance(
        result["time_in_range"], PatientType.ADULT
    )
    print(f"TIR Acceptable: {count}/{len(results)} categories within range")
    print(f"Details: {results}")

    print("done")
