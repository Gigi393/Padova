from simglucose.patient.t1dpatient import T1DPatient, Action
from simglucose.patient.t1dpatient_2 import CtrlObservation
from simglucose.controller.basal_bolus_ctrller import BBController

from datetime import datetime, timedelta
import logging
from pathlib import Path
from plotting import plot_and_show

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


patient_name = "adolescent#003"
p = T1DPatient.withName(patient_name)
basal = p._params.u2ss * p._params.BW / 6000  # U/min


t = []
CHO = []
insulin = []
BG = []


copy_state = 50

ctrl = BBController()
logger.info("Using Basal-Bolus Controller")

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
        meal=carb,
        patient_name=patient_name,
        sample_time=p.sample_time,
    )
    ins = ctrl_action.basal + ctrl_action.bolus

    act = Action(insulin=ins, CHO=carb)

    t.append(p.t)
    CHO.append(act.CHO)
    insulin.append(act.insulin)
    BG.append(p.observation.Gsub)
    p.step(act)


plot_and_show(t, BG, CHO, insulin, ctrl.target, "bb_controller")
