import logging
from datetime import datetime
from pathlib import Path
from collections import namedtuple
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero_with_meal_bolus import (
    ORefZeroWithMealBolus,
    CtrlObservation,
)
from simglucose.simulation.scenario_simple import Scenario
from analytics import TIRConfig
from plotting import plot_and_save_with_tir
from bg_attacker import BGAttacker


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
test_patient_dir = img_dir / "test_attack" / "attack_oref_zero_bolus"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_save_dir = test_patient_dir / timestamp
image_save_dir.mkdir(parents=True, exist_ok=True)


def patient_oref0_with_meal_bolus(
    patient_name="adolescent#003",
    img_save_dir=image_save_dir,
    attack_step: float = 1.8,  # mg/dL per minute
    attack_maintain: float = 60,  # minutes
    profile=None,
    attacking=True,
    save_fig=False,
    meal_time=360,  # Single meal at 6 hours (360 minutes)
    meal_amount=50,  # 50g carbs
    release_time_before_meal=10,  # minutes before meal to release bolus
    carb_estimation_error=0.3,  # +/- percentage of carb estimation error
    start_attack_time = 180, #min
):

    p = T1DMPatient.withName(patient_name)
    if profile is not None:
        profile["carb_ratio"] = p.carb_ratio
        profile["current_basal"] = p.basal * 60  # U/min to U/h

    # Single meal schedule - one meal per day
    meal_schedule = [(meal_time, meal_amount)]
    logger.info(f"Meal schedule: {meal_schedule}")

    # Initialize combined controller with meal bolus support
    ctrl = ORefZeroWithMealBolus(
        patient_name=patient_name,
        server_url="http://localhost:3000",
        timeout=30000,  # TODO: remove timeout when not in debug
        profile=profile,
        meal_schedule=meal_schedule,
        carb_factor=(
            profile["carb_ratio"] if profile and "carb_ratio" in profile else 10
        ),
        release_time_before_meal=release_time_before_meal,
        carb_estimation_error=carb_estimation_error,
        t_start=p.t_start,
    )
    ctrl.initialize()

    t = []
    CHO = []
    insulin = []
    BG = []

    attacker = BGAttacker(step=attack_step, maintain_duration=attack_maintain)
    attack_started = False  # Track if attack has been triggered

    max_t = 12 * 60  # Simulate for 12 hours
    while p.t_elapsed < max_t:
        carb = meal_amount if p.t_elapsed == meal_time else 0

        # if p.observation.Gsub < 39:
        #     print("Patient is dead")
        #     break

        # Start attack when CGM reaches 200 mg/dL
        if not attack_started and p.t_elapsed>start_attack_time:
            print(f"Starting attack at t={p.t_elapsed}, CGM={p.observation.Gsub:.1f}")
            attacker.start_attack(p.t_elapsed, p.observation.Gsub)
            attack_started = True

        if attacking:
            glucose = attacker.get_spoofed_bg(p.t_elapsed, p.observation.Gsub)
        else:
            glucose = p.observation.Gsub

        # Simple observation with just CGM - ORefZeroWithMealBolus handles bolus internally
        ctrl_obs = CtrlObservation(CGM=glucose)

        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            meal=carb,
            time=p.t,
        )

        ins = ctrl_action.basal + ctrl_action.bolus
        act = Action(insulin=ins, CHO=carb)  # U/min

        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(glucose)
        p.step(act)

        print(
            f"\033[94mt: {p.t_elapsed}, t: {p.t} BG: {glucose}, CHO: {carb}, Insulin: {ins}\033[0m"
        )

    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(BG)

    sanitized_patient_name = patient_name.replace("#", "_")
    attack_suffix = "with_attack" if attacking else "no_attack"
    fig_title = f"test_patient_{sanitized_patient_name}_{meal_amount}_at_{meal_time}min_{attack_suffix}"
    if save_fig:
        file_name = img_save_dir / f"{fig_title}.png"
        plot_and_save_with_tir(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_bg,
            file_name,
            time_in_range,
            tir_config,
        )

    return time_in_range


if __name__ == "__main__":
    """
    using android app logic from
    https://github.com/ImperialGlobalSingapore/GoodV/blob/135890492f03820744cdf6efe998eceaa9da4721/app/src/main/java/com/kk4vcz/goodv/CGM_RW_Fragment.java#L282-L350

    to attack the patient CGM readings and see if OpenAPS can be tricked into delivering more insulin
    """

    # patient_name = "adult#007"
    # profile = {
    #     "sens": 50,
    #     "dia": 8,
    #     "max_iob": 25,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
    #     "max_basal": 4,  # from paper, max 10
    #     "max_bg": 180,
    #     "min_bg": 90,
    # }

    patient_name = "child#002"
    profile = {
        "sens": 50,
        "dia": 8,
        "max_iob": 15,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 4,  # from paper, max 10
        "max_bg": 180,
        "min_bg": 90,
    }
    
    attack_step = 1.8  # mg/dL per minute
    attack_maintain = 30  # minutes
    meal_time = 20  # Single meal at 20 minutes
    meal_amount = 75  # 75g carbs
    release_time_before_meal = 10
    carb_estimation_error = 0.3

    # Run without attack
    logger.info("Running simulation WITHOUT attack")
    tir_without_attack = patient_oref0_with_meal_bolus(
        patient_name=patient_name,
        profile=profile,
        save_fig=True,
        attacking=False,
        attack_step=attack_step,
        attack_maintain=attack_maintain,
        meal_time=meal_time,
        meal_amount=meal_amount,
        release_time_before_meal=release_time_before_meal,
        carb_estimation_error=carb_estimation_error,
    )

    # Run with attack
    logger.info("Running simulation WITH attack")
    tir_with_attack = patient_oref0_with_meal_bolus(
        patient_name=patient_name,
        profile=profile,
        save_fig=True,
        attacking=True,
        attack_step=attack_step,
        attack_maintain=attack_maintain,
        meal_time=meal_time,
        meal_amount=meal_amount,
        release_time_before_meal=release_time_before_meal,
        carb_estimation_error=carb_estimation_error,
        start_attack_time=200
    )

    # Print comparison
    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Time in Range WITH attack: {tir_with_attack}")
    logger.info(f"Time in Range WITHOUT attack: {tir_without_attack}")
    logger.info("=" * 60)
