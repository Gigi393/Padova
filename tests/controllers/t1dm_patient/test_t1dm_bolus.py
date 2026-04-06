import logging
from pathlib import Path

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.meal_bolus_ctrller import MealAnnouncementBolusController
from simglucose.simulation.scenario_simple import Scenario
from analytics import TIRConfig
from plotting import plot_and_show_with_tir

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def scenario_to_meal_schedule(scenario, body_weight=None):
    """
    Convert a Scenario enum to a meal schedule list of tuples.

    Args:
        scenario: Scenario enum
        body_weight: Patient body weight in kg (optional)

    Returns:
        List of tuples (time_minutes, carbs_grams)
    """
    meal_schedule = []
    for t in range(0, scenario.max_t + 1):
        carbs = scenario.get_carb(t, body_weight)
        if carbs > 0:
            meal_schedule.append((t, carbs))
    return meal_schedule


def run_patient_with_bolus_only(
    patient_name="adult#001",
    scenario=Scenario.ONE_DAY,
    profile=None,
    show_plot=True,
    release_time_before_meal=10,  # minutes before meal to release bolus
    carb_estimation_error=0.3,  # +/- percentage of carb estimation error
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

    # Convert scenario to meal schedule
    meal_schedule = scenario_to_meal_schedule(scenario, p.body_weight)
    logger.info(f"Meal schedule: {meal_schedule}")

    # Initialize controller
    meal_bolus_ctrl = MealAnnouncementBolusController(
        meal_schedule=meal_schedule,
        carb_factor=(
            profile["carb_ratio"] if profile and "carb_ratio" in profile else 10
        ),
        release_time_before_meal=release_time_before_meal,
        carb_estimation_error=carb_estimation_error,
        t_start=p.t_start,  # Pass patient start time for elapsed time calculation
    )

    basal = p.basal  # U/min

    # Storage for simulation data
    t = []
    CHO = []
    insulin = []
    BG = []

    # Run simulation
    logger.info(f"Starting simulation with scenario: {scenario.name}")
    while p.t_elapsed < scenario.max_t:
        # Get meal for current time
        carb = scenario.get_carb(p.t_elapsed, p.body_weight)

        # Get bolus from meal bolus controller
        bolus_action = meal_bolus_ctrl.policy(p.t)  # Pass datetime, controller will calculate elapsed

        # Check for severe hypoglycemia
        if p.observation.Gsub < 39:
            logger.error("Severe hypoglycemia detected - stopping simulation")

        ins = basal + bolus_action.bolus  # Total insulin (basal + bolus)
        act = Action(insulin=ins, CHO=carb)
        # Record data
        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
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
        plot_and_show_with_tir(
            t,
            BG,
            CHO,
            insulin,
            110,
            f"T1DM Patient {patient_name} with Meal Bolus - {scenario.name}",
            time_in_range,
            tir_config,
        )

    return time_in_range


if __name__ == "__main__":
    custom_profile = {
        "sens": 45,
        "dia": 7.0,
        "carb_ratio": 20,
        "max_iob": 12,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 4,  # from paper, max 10
        "max_daily_basal": 0.9,  # from paper
        "max_bg": 140,
        "min_bg": 90,
        "maxCOB": 120,  # from oref0 code
        "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 60}]},
        "min_5m_carbimpact": 8,  # from paper and oref0 code
    }

    run_patient_with_bolus_only(
        patient_name="adult#007",
        scenario=Scenario.ONE_DAY,
        profile=custom_profile,
        show_plot=True,
        release_time_before_meal=10,
        carb_estimation_error=0.3,
    )
