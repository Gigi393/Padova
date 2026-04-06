import logging
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec

# matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero_with_meal_bolus import (
    ORefZeroWithMealBolus,
    CtrlObservation,
)
from analytics import TIRConfig, TIRCategory


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
test_patient_dir = img_dir / "test_attack" / "low_glucose_attack"
image_save_dir = test_patient_dir
image_save_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    """
    using android app logic from
    https://github.com/ImperialGlobalSingapore/GoodV/blob/135890492f03820744cdf6efe998eceaa9da4721/app/src/main/java/com/kk4vcz/goodv/CGM_RW_Fragment.java#L282-L350

    to attack the patient CGM readings and see if OpenAPS can be tricked into delivering more insulin
    """

    patient_name = "adult#007"
    profile = {
        "sens": 50,
        "dia": 8,
        "max_iob": 25,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
        "max_basal": 4,  # from paper, max 10
        "max_bg": 180,
        "min_bg": 90,
    }

    # patient_name = "child#002"
    # profile = {
    #     "sens": 50,
    #     "dia": 8,
    #     "max_iob": 15,  # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
    #     "max_basal": 4,  # from paper, max 10
    #     "max_bg": 180,
    #     "min_bg": 90,
    # }

    release_time_before_meal = 10
    carb_estimation_error = 0.3
    max_t = 48 * 60  # Simulate for 48 hours
    start_attack_time = 24 * 60  # Start attack after 24 hours
    low_glucose_bound = (60, 100)  # Target low glucose between 60-80 mg/dL
    attack = False

    ###############################################################################

    p = T1DMPatient.withName(patient_name)
    if profile is not None:
        profile["carb_ratio"] = p.carb_ratio
        profile["current_basal"] = p.basal * 60  # U/min to U/h

    # Single meal schedule - one meal per day
    meal_time = [7, 12, 19, 7 + 24, 12 + 24, 19 + 24]  # in hours
    meal_amount = [40, 60, 70, 40, 60, 70]  # in grams
    meal_schedule = [(t * 60, a) for t, a in zip(meal_time, meal_amount)]
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
    REAL_BG = []
    BASAL = []
    BOLUS = []
    OB_BG = []

    while p.t_elapsed < max_t:
        carb = 0
        for time_point, amount in meal_schedule:
            if p.t_elapsed == time_point:
                carb = amount
                break
        # if p.observation.Gsub < 39:
        #     print("Patient is dead")
        #     break

        # Start attack when CGM reaches 200 mg/dL
        if attack and p.t_elapsed > start_attack_time:
            glucose = random.uniform(low_glucose_bound[0], low_glucose_bound[1])
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
        OB_BG.append(glucose)
        REAL_BG.append(p.observation.Gsub)
        BASAL.append(ctrl_action.basal)
        BOLUS.append(ctrl_action.bolus)
        CHO.append(act.CHO)
        p.step(act)

        print(
            f"\033[94mt: {p.t_elapsed}, t: {p.t} BG: {glucose}, CHO: {carb}, Insulin: {ins}\033[0m"
        )

    tir_config = TIRConfig()  # Defaults to BASIC standard
    time_in_range = tir_config.calculate_time_in_range(REAL_BG)

    sanitized_patient_name = patient_name.replace("#", "_")
    attack_suffix = "low_glucose_attack"
    fig_title = f"test_patient_{sanitized_patient_name}_{meal_amount}_at_{meal_time}min_{attack_suffix}"
    file_name = image_save_dir / f"{fig_title}.png"

    # Create figure with gridspec layout for TIR scale
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(
        3, 2, width_ratios=[15, 1], height_ratios=[1, 1, 1], wspace=0.1
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    scale_ax = fig.add_subplot(gs[:, 1])

    # Plot BG
    ax0.plot(t, OB_BG, label="Observed BG")
    ax0.plot(t, REAL_BG, label="Real BG")
    ax0.plot(t, [ctrl.target_bg] * len(t), "r--", label="Target BG")
    ax0.plot(t, [70] * len(t), "b--", label="Hypoglycemia")
    ax0.plot(t, [180] * len(t), "k--", label="Hyperglycemia")
    ax0.grid()
    ax0.set_ylabel("BG (mg/dL)")
    ax0.legend(loc="upper right")

    # Plot CHO
    ax1.plot(t, CHO)
    ax1.grid()
    ax1.set_ylabel("CHO (g)")

    # Plot Basal on primary y-axis
    basal_color = "#1580aa"  # Teal
    line1 = ax2.plot(t, BASAL, color=basal_color, label="Basal")
    ax2.set_ylabel("Basal (U/min)", color=basal_color)
    ax2.tick_params(axis="y", labelcolor=basal_color)
    ax2.grid()
    ax2.set_xlabel("Time (min)")

    # Plot Bolus on secondary y-axis
    bolus_color = "#ea580c"  # Orange
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(t, BOLUS, color=bolus_color, label="Bolus")
    ax2_twin.set_ylabel("Bolus (U/min)", color=bolus_color)
    ax2_twin.tick_params(axis="y", labelcolor=bolus_color)

    # Combined legend
    lines = line1 + line2
    labels = [str(line.get_label()) for line in lines]
    ax2.legend(lines, labels, loc="upper right")
    fig.suptitle(f"{fig_title}")
    fig.tight_layout()

    # Plot time in range scale bar
    scale_ax.axis("off")
    order = tir_config.get_order()
    thresholds = tir_config.get_thresholds()
    x_position = 0
    bar_width = 0.8
    bottom = 0

    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]
            if percentage > 0:
                color = tir_config.get_color(category)
                scale_ax.bar(
                    x_position,
                    percentage,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=1,
                )
                scale_ax.text(
                    x_position,
                    bottom + percentage / 2,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )
                bottom += percentage

    cumulative_height = 0
    for category in order:
        if category in time_in_range:
            percentage = time_in_range[category]
            if percentage > 0:
                if category != TIRCategory.VERY_LOW:
                    scale_ax.text(
                        x_position - bar_width / 2 - 0.1,
                        cumulative_height,
                        f"{thresholds[category]}",
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
                cumulative_height += percentage
                next_key_index = order.index(category) + 1
                if next_key_index < len(order):
                    next_category = order[next_key_index]
                    scale_ax.text(
                        x_position - bar_width / 2 - 0.1,
                        cumulative_height,
                        f"{thresholds[next_category]}",
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

    scale_ax.set_xlim(-0.5, 0.5)
    scale_ax.set_ylim(0, 100)
    scale_ax.set_title("Time in Range", fontsize=12)

    # Save and close
    # fig.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0.1)
    # plt.close(fig)
    plt.show()
