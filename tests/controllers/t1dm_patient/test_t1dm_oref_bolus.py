"""
Simple example demonstrating how to use T1DM patient with ORef0 controller.

This script shows a basic simulation of a T1DM patient using the OpenAPS ORef0
algorithm for automated insulin delivery. It's meant as a reference implementation
and testing example.
"""

import logging

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero_with_meal_bolus import (
    ORefZeroWithMealBolus,
    CtrlObservation,
)
from analytics import TIRConfig
from plotting import plot_bg_cho_iob_and_show_with_tir

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def run_patient_with_oref0_bolus(
    patient_name="adult#001",
    profile=None,
    show_plot=True,
    meal_time=360,  # Single meal at 6 hours (360 minutes)
    meal_amount=50,  # 50g carbs
    release_time_before_meal=10,  # minutes before meal to release bolus
    carb_estimation_error=0.3,  # +/- percentage of carb estimation error
    simulation_time=720,  # 12 hours
):
    """
    Run a simulation of a T1DM patient with ORef0 + Meal Bolus controller.

    Args:
        patient_name: Name of the patient (e.g., "adult#001", "child#001")
        profile: Optional ORef0 profile dict with parameters like sens, dia, carb_ratio, etc.
        show_plot: Whether to display the results plot
        meal_time: Time in minutes when meal is delivered
        meal_amount: Amount of carbs in grams
        release_time_before_meal: Minutes before meal to release bolus
        carb_estimation_error: Percentage of error in carb estimation
        simulation_time: Total simulation time in minutes

    Returns:
        dict: Time in range statistics
    """
    # Initialize patient
    p = T1DMPatient.withName(patient_name)
    logger.info(f"Patient {patient_name} initialized")
    if profile is not None:
        profile["carb_ratio"] = p.carb_ratio
        profile["current_basal"] = p.basal * 60  # U/min to U/h

    # Single meal schedule
    meal_schedule = [(meal_time, meal_amount)]
    logger.info(f"Meal schedule: {meal_schedule}")

    # Initialize combined controller
    combined_ctrl = ORefZeroWithMealBolus(
        patient_name=patient_name,
        server_url="http://localhost:3000",
        timeout=3000,  # TODO: DEBUG only
        profile=profile,
        meal_schedule=meal_schedule,
        carb_factor=(
            profile["carb_ratio"] if profile and "carb_ratio" in profile else 10
        ),
        release_time_before_meal=release_time_before_meal,
        carb_estimation_error=carb_estimation_error,
        t_start=p.t_start,  # Pass patient start time for meal bolus calculation
    )

    # Initialize patient on the controller
    if not combined_ctrl.initialize():
        raise ValueError("Failed to initialize ORefZero controller")

    logger.info("ORefZeroWithMealBolus controller initialized")

    # Storage for simulation data
    t = []
    CHO = []
    IOB = []
    BG = []

    # Run simulation
    logger.info(f"Starting simulation for {simulation_time} minutes")
    while p.t_elapsed < simulation_time:
        # Deliver meal at meal_time
        carb = meal_amount if p.t_elapsed == meal_time else 0

        # Create observation for controller (just CGM for ORefZeroWithMealBolus)
        ctrl_obs = CtrlObservation(CGM=p.observation.Gsub)

        # Check for severe hypoglycemia
        if p.observation.Gsub < 39:
            logger.error("Severe hypoglycemia detected - stopping simulation")
            break

        # Get combined controller action (ORefZero basal + meal bolus)
        combined_action = combined_ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            meal=carb,
            time=p.t,  # datetime for both controllers (meal_bolus calculates elapsed from t_start)
        )

        # Calculate total insulin (basal + bolus)
        ins = combined_action.basal + combined_action.bolus  # U/min
        act = Action(insulin=ins, CHO=carb)

        # Record data
        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        IOB.append(p.get_iob())
        BG.append(p.observation.Gsub)

        # Step the patient simulation
        p.step(act)

        # Log progress (every hour)
        if p.t_elapsed % 60 == 0:
            logger.info(
                f"Time: {p.t_elapsed}min, BG: {p.observation.Gsub:.1f} mg/dL, "
                f"CHO: {carb}g, Insulin: {ins:.3f} U/min"
            )

    # Calculate time in range statistics
    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(BG)

    logger.info("\n=== Time in Range Results ===")
    for category, percentage in time_in_range.items():
        logger.info(f"{category.value}: {percentage:.1f}%")

    # Display plot if requested
    if show_plot:
        plot_bg_cho_iob_and_show_with_tir(
            t,
            BG,
            CHO,
            IOB,
            combined_ctrl.target_bg,
            f"T1DM Patient {patient_name} with ORef0 + Meal Bolus - {meal_amount}g at {meal_time}min",
            time_in_range,
            tir_config,
        )

    return time_in_range


if __name__ == "__main__":
    # Example: Run with custom profile
    # refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    print("\n" + "=" * 60)
    print("Example: Adult patient with custom ORef0 profile")
    print("=" * 60)

    # adult#007
    patient_name = "adult#007"
    custom_profile = {
        "sens": 50,
        "dia": 8.0,
        "carb_ratio": 20,  # changed later from patient
        "max_iob": 27.5,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 4,  # from paper, max 10
        "max_bg": 180,
        "min_bg": 70,
    }

    # child#002
    # patient_name = "child#002"
    # custom_profile = {
    #     "sens": 50,
    #     "dia": 8.0,
    #     "carb_ratio": 30,  # changed later from patient
    #     "max_iob": 15,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
    #     "max_basal": 4,  # from paper, max 10
    #     "max_bg": 180,
    #     "min_bg": 90,
    # }

    run_patient_with_oref0_bolus(
        patient_name=patient_name,
        profile=custom_profile,
        show_plot=True,
        meal_time=20,  # Meal at 6 hours
        meal_amount=75,  # 50g carbs
        release_time_before_meal=10,
        carb_estimation_error=0,  # 30% carb estimation error
        simulation_time=720,  # 12 hours
    )
