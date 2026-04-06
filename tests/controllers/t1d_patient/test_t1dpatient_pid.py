import logging
from pathlib import Path


from simglucose.patient.t1dpatient import T1DPatient, Action
from simglucose.patient.t1dpatient_2 import CtrlObservation
from simglucose.controller.simple_pid_ctrller import SimplePIDController

from plotting import plot_and_show, plot_and_save
from analytics import get_rmse

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
# parent folder
parent_folder = file_path.parent

log_dir = parent_folder / "logs"
img_dir = parent_folder / "imgs"

log_dir.mkdir(exist_ok=True)
img_dir.mkdir(exist_ok=True)


# this pid controller is too complex to fix, ignore now"
""""
def run_sim_pid(k_P, k_I, k_D):
    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = PIDController(k_P=0.001, k_I=0.00001, k_D=0.001)
    logger.info("Using PID Controller")

    while p.t < 2000:
        carb = 0

        if p.t == 100:
            carb = 80

        # if p.t == 200:
        #     carb = 50

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )
        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    save_to_csv(t, BG, CHO, insulin, k_P, k_I, k_D)
"""


def save_name_pattern(k_P, k_I, k_D, sample_time, basal_rate, remark=""):
    return f"{remark}pid_st{sample_time}_br{basal_rate}_p{k_P}_i{k_I}_d{k_D}"


def run_sim_simple_pid_no_meal(
    k_P,
    k_I,
    k_D,
    sample_time=5,
    basal_rate=0.2,
    patient_name="adolescent#003",
    sim_time=2000,
    save_fig=False,
    show_fig=False,
    log=True,
    img_folder_name="no_meal",
):
    no_meal_log_dir = log_dir / img_folder_name
    no_meal_img_dir = img_dir / img_folder_name
    no_meal_log_dir.mkdir(exist_ok=True, parents=True)
    no_meal_img_dir.mkdir(exist_ok=True, parents=True)

    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(
        k_P=k_P,
        k_I=k_I,
        k_D=k_D,
        sampling_time=sample_time,
        basal_rate=basal_rate,
    )
    if log:
        logger.info(
            f"Testing {patient_name} using Simple PID Controller p{k_P}, i{k_I}, d{k_D}, br{basal_rate} without meal"
        )

    while p.t < sim_time:

        carb = 0

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )
        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark=f"{patient_name}_simple_"
    )
    if save_fig:
        plot_and_save(
            t, BG, CHO, insulin, ctrl.target_BG, no_meal_img_dir / f"{file_name}.png"
        )
    if show_fig:
        plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)

    rmse = get_rmse(BG, ctrl.target_BG)
    if log:
        logger.info(f"{patient_name}: {k_P}, {k_I}, {k_D} rmse: {rmse}")
    return rmse


def run_sim_simple_pid_single_meal(
    k_P,
    k_I,
    k_D,
    sample_time=5,
    basal_rate=0.2,
    patient_name="adolescent#003",
    sim_time=2000,
    save_fig=False,
    show_fig=False,
    log=True,
    img_folder_name="single_meal",
):
    single_meal_log_dir = log_dir / img_folder_name
    single_meal_img_dir = img_dir / img_folder_name
    single_meal_log_dir.mkdir(exist_ok=True, parents=True)
    single_meal_img_dir.mkdir(exist_ok=True, parents=True)

    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(
        k_P=k_P, k_I=k_I, k_D=k_D, sampling_time=sample_time, basal_rate=basal_rate
    )
    if log:
        logger.info(
            f"Testing {patient_name} using Simple PID Controller p{k_P}, i{k_I}, d{k_D}, br{basal_rate} with single meal"
        )

    while p.t < sim_time:
        carb = 0

        if p.t == 100:
            carb = 80

        # if p.t == 200:
        #     carb = 50

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )

        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark=f"{patient_name}_simple_"
    )
    if save_fig:
        plot_and_save(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_BG,
            single_meal_img_dir / f"{file_name}.png",
        )
    if show_fig:
        plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)

    rmse = get_rmse(BG, ctrl.target_BG)
    if log:
        logger.info(f"{patient_name}: {k_P}, {k_I}, {k_D} rmse: {rmse}")
    return rmse


def run_sim_simple_pid_attack(
    k_P,
    k_I,
    k_D,
    sample_time=5,
    basal_rate=0.2,
    patient_name="adolescent#003",
    sim_time=2000,
    save_fig=False,
    show_fig=False,
    log=True,
    img_folder_name="attack_no_meal",
):
    attack_no_meal_log_dir = log_dir / img_folder_name
    attack_no_meal_img_dir = img_dir / img_folder_name
    attack_no_meal_log_dir.mkdir(exist_ok=True, parents=True)
    attack_no_meal_img_dir.mkdir(exist_ok=True, parents=True)

    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    ctrl = SimplePIDController(
        k_P=k_P, k_I=k_I, k_D=k_D, sampling_time=sample_time, basal_rate=basal_rate
    )
    if log:
        logger.info(
            f"Testing {patient_name} using Simple PID Controller p{k_P}, i{k_I}, d{k_D}, br{basal_rate} without meal"
        )

    while p.t < sim_time:
        carb = 0

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        if p.t == 100:
            ctrl_obs = CtrlObservation(300)
            if log:
                logger.info(f"Attack at {p.t} min with {ctrl_obs.CGM} CGM")

        ctrl_action = ctrl.policy(
            observation=ctrl_obs,
            reward=0,
            done=False,
            patient_name=patient_name,
            meal=carb,
            time=p.t,
        )

        ins = ctrl_action.basal + ctrl_action.bolus

        act = Action(insulin=ins, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    if log:
        logger.info("Simulation done")

    file_name = save_name_pattern(
        k_P, k_I, k_D, sample_time, ctrl.basal_rate, remark=f"{patient_name}_simple_"
    )
    if save_fig:
        plot_and_save(
            t,
            BG,
            CHO,
            insulin,
            ctrl.target_BG,
            attack_no_meal_img_dir / f"{file_name}.png",
        )
    if show_fig:
        plot_and_show(t, BG, CHO, insulin, ctrl.target_BG, file_name)

    rmse = get_rmse(BG, ctrl.target_BG)
    if log:
        logger.info(f"{patient_name}: {k_P}, {k_I}, {k_D} rmse: {rmse}")
    return rmse


if __name__ == "__main__":
    # run_sim_simple_pid_no_meal(
    #     k_P=0.001, k_I=0.00001, k_D=0.001, sample_time=5, basal_rate=0.2
    # )

    # best params
    # k_P=7.5e-05, k_I=1e-09, k_D=0, sample_time=5, basal_rate=0.06
    # run for all patients
    # patients = get_patients()
    # for patient in patients:
    #     run_sim_simple_pid_no_meal(
    #         k_P=7.5e-05,
    #         k_I=1e-09,
    #         k_D=0,
    #         sample_time=5,
    #         basal_rate=0.06,
    #         patient_name=patient,
    #         save_fig=True,
    #     )

    # best params by group
    # group: adolescent
    # k_p: 1.0e-03
    # k_i: 6.2e-07 => 1e-6
    # k_d: 8.0e-02 =>0.1
    # basal_rate: 7.2e-02 =>0.1
    # group: adult
    # k_p: 1.0e-03
    # k_i: 9.1e-07 =>1e-6
    # k_d: 3.0e-02 =>0.1
    # basal_rate: 9.8e-02 =>0.1
    # group: child
    # k_p: 8.2e-04 =>0.001
    # k_i: 6.4e-07 =>1e-6
    # k_d: 8.0e-02 =>0.1
    # basal_rate: 3.1e-02 =>0.03

    # best for patient adolescent#003
    # run_sim_simple_pid_no_meal(
    #     k_P=0.001,
    #     k_I=1e-06,
    #     k_D=0.1,
    #     sample_time=5,
    #     basal_rate=0.05,
    #     patient_name="adolescent#003",
    #     save_fig=False,
    #     show_fig=True,
    # )

    # run_sim_simple_pid_attack(
    #     k_P=0.001,
    #     k_I=1e-06,
    #     k_D=0.1,
    #     sample_time=5,
    #     basal_rate=0.05,
    #     patient_name="adolescent#003",
    #     save_fig=False,
    #     show_fig=True,
    #     log=True,
    # )

    # best for patient adolescent#003
    run_sim_simple_pid_single_meal(
        k_P=0.001,
        k_I=1e-06,
        k_D=0.1,
        sample_time=5,
        basal_rate=0.05,
        patient_name="adolescent#003",
        save_fig=False,
        show_fig=True,
        log=True,
    )
