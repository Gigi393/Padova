from collections import namedtuple
import pandas as pd
import numpy as np
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from matplotlib import pyplot as plt
import json

CtrlObservation = namedtuple("CtrlObservation", ["CGM"])
Action = namedtuple("patient_action", ["CHO", "insulin"])
PatientObservation = namedtuple("observation", ["Gsub"])

def run_original_simulation():
    patient_name = "adolescent#003"
    p = T1DPatient.withName(patient_name, seed=42)
    
    t = []
    CHO = []
    insulin = []
    BG = []
    states = []
    ctrl = BBController(target=140)
    
    while p.t < 50:
        carb = 0
        if p.t == 5:
            carb = 20
            
        ctrl_obs = CtrlObservation(p.observation.Gsub)
        ctrl_action = ctrl.policy(
            observation=ctrl_obs, 
            reward=0, 
            done=False, 
            patient_name=patient_name, 
            meal=carb
        )
        
        ins = ctrl_action.basal + ctrl_action.bolus
        print(f"Time: {p.t}, BG: {p.observation.Gsub:.2f}, Insulin: {ins:.4f}, Carbs: {carb}")
        
        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)
        
        states.append({
            "params": p._params.to_dict(),
            "state": p.state.tolist(),
            "seed": p._seed,
            "t": p.t,
            "planned_meal": p.planned_meal,
            "is_eating": p.is_eating,
            "glucose": p.observation.Gsub,
            "insulin": ins,
        })

    # Save data
    with open("sim.json", "w") as f:
        json.dump({"bg": BG, "cho": CHO, "insulin": insulin, "states": states}, f)
    
    # Plot
    fig, ax = plt.subplots(3, sharex=True, figsize=(10, 8))
    ax[0].plot(t, BG)
    ax[0].set_ylabel("Blood Glucose")
    ax[0].grid(True)
    
    ax[1].plot(t, CHO)
    ax[1].set_ylabel("CHO")
    ax[1].grid(True)
    
    ax[2].plot(t, insulin)
    ax[2].set_ylabel("Insulin")
    ax[2].set_xlabel("Time")
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("original_sim.png")
    print("Original simulation completed and saved to sim.json")
    print("Plot saved as original_sim.png")

if __name__ == "__main__":
    run_original_simulation()
