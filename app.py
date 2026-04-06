from collections import namedtuple
from uuid import uuid4
from typing import Dict, Optional, Any
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.oref_zero import CtrlObservation
from simglucose.controller.oref_zero_with_meal_bolus import ORefZeroWithMealBolus
from simglucose.patient.t1dm_patient import (
    T1DMPatient,
    Action as PatientAction,
    Observation as PatientObservation,
)
from simglucose.controller.base import Controller

# Define the namedtuples as in the original code

# Controller mapping
controller_map = {
    "basal_bolus": BBController,
    "pid": PIDController,
    "openaps": ORefZeroWithMealBolus,
}


# Pydantic models for request/response validation
class InitRequest(BaseModel):
    patient: str = "adolescent#003"
    controller_algorithm: str = "basal_bolus"
    controller_kwargs: Optional[Dict[str, Any]] = None


class InitResponse(BaseModel):
    initial_glucose: float
    patient_id: str


class StepRequest(BaseModel):
    glucose_reading: float
    carbs: int = 0
    delta_time: int = 1
    pump: str = "DefaultPump"
    attack_glucose: Optional[float] = (
        None  # Added: fake glucose data for attack scenario
    )


class StepResponse(BaseModel):
    glucose: float
    insulin: float
    attack_scenario: bool = False  # Added: flag for attack scenario
    real_glucose: Optional[float] = None  # Added: real glucose value
    attack_glucose: Optional[float] = None  # Added: attack glucose value
    patient_iob: Optional[float] = None  # Added: IOB from patient physiological model
    openaps_iob: Optional[float] = None  # Added: IOB from OpenAPS algorithm


# Create FastAPI app
app = FastAPI(title="Glucose Simulation API")


# Store patients and their controllers in memory (consider a database for production)
class PatientSession:
    def __init__(self, patient: T1DMPatient, controller: Controller):
        self.patient = patient
        self.controller = controller


patient_map: Dict[str, PatientSession] = {}

# Define logger
logger = logging.getLogger(__name__)


@app.post("/init", response_model=InitResponse)
def init(request: InitRequest):
    """Initialize a new patient simulation with controller"""
    t1dm_patient = T1DMPatient.withName(request.patient)

    # Create controller based on algorithm
    if request.controller_algorithm == "openaps":
        # ORefZeroWithMealBolus needs current_basal, profile, meal_schedule, and t_start
        controller_profile = {}
        if request.controller_kwargs:
            controller_profile = request.controller_kwargs.get("profile", {})
        controller_profile["carb_ratio"] = t1dm_patient.carb_ratio
        controller_profile["current_basal"] = t1dm_patient.basal * 60

        # Get meal schedule and other parameters from controller_kwargs
        meal_schedule = (
            request.controller_kwargs.get("meal_schedule", [])
            if request.controller_kwargs
            else []
        )
        release_time_before_meal = (
            request.controller_kwargs.get("release_time_before_meal", 10)
            if request.controller_kwargs
            else 10
        )
        carb_estimation_error = (
            request.controller_kwargs.get("carb_estimation_error", 0.3)
            if request.controller_kwargs
            else 0.3
        )

        ctrl = ORefZeroWithMealBolus(
            patient_name=request.patient,
            server_url=os.getenv("OPENAPS_URL", "http://localhost:3000"),
            profile=controller_profile,
            meal_schedule=meal_schedule,
            carb_factor=controller_profile["carb_ratio"],
            release_time_before_meal=release_time_before_meal,
            carb_estimation_error=carb_estimation_error,
            t_start=t1dm_patient.t_start,
        )
        ctrl.initialize()

    elif request.controller_algorithm in controller_map:
        # Pass controller_kwargs if provided
        if request.controller_kwargs:
            ctrl = controller_map[request.controller_algorithm](
                **request.controller_kwargs
            )

        else:
            ctrl = controller_map[request.controller_algorithm]()

    else:
        raise HTTPException(status_code=400, detail="Unsupported controller algorithm")

    # Generate a unique ID for this patient
    patient_id = str(uuid4())

    # Store the patient session (patient + controller)
    patient_map[patient_id] = PatientSession(t1dm_patient, ctrl)

    if request.controller_algorithm == "openaps":
        profile = ctrl.get_profile()
        if not profile:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve patient profile from OpenAPS",
            )

        max_iob = profile.get("max_iob", None)
        return {
            "initial_glucose": t1dm_patient.observation.Gsub,
            "patient_id": patient_id,
            "max_iob": max_iob,
        }

    return {
        "initial_glucose": t1dm_patient.observation.Gsub,
        "patient_id": patient_id,
    }


@app.post("/step/{patient_id}", response_model=StepResponse)
def step(patient_id: str, request: StepRequest):
    """Take a simulation step for a specific patient"""
    # Check if patient exists
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")

    session = patient_map[patient_id]
    patient = session.patient
    ctrl = session.controller

    # Determine if this is an attack scenario
    is_attack = request.attack_glucose is not None

    # Create controller observation - if attack scenario, use fake glucose data
    glucose_for_controller = (
        request.attack_glucose if is_attack else request.glucose_reading
    )
    # print glucose reading from the phone
    print(
        f"Glucose reading for controller: {request.glucose_reading}, is attack: {is_attack}, Attack glucose: {request.attack_glucose}"
    )
    ctrl_obs = CtrlObservation(
        glucose_for_controller, bolus=0
    )  # bolus handle inside controller

    # Get controller action - reuse the same controller instance
    ctrl_action = ctrl.policy(
        observation=ctrl_obs,
        reward=0,
        done=False,
        meal=request.carbs,
        time=patient.t,
    )

    # Calculate insulin
    insulin = ctrl_action.basal + ctrl_action.bolus

    # Create patient action
    patient_action = PatientAction(insulin=insulin, CHO=request.carbs)

    # Update patient state
    patient.step(patient_action)

    # Get new glucose level
    glucose = patient.observation.Gsub

    # Get IOB from patient model (physiological)
    # Use subtract_baseline=False to match OpenAPS comparison
    patient_iob_value = patient.get_iob(include_plasma=True, subtract_baseline=False)
    patient_iob_value = patient_iob_value if patient_iob_value is not None else 0.0

    # Get IOB from OpenAPS controller (if using openaps)
    openaps_iob_value = 0.0
    if isinstance(ctrl, ORefZeroWithMealBolus):
        openaps_iob_data = ctrl.get_iob()
        openaps_iob_value = openaps_iob_data["iob_value"] if openaps_iob_data else 0.0

    # Build response
    response = {
        "glucose": glucose,
        "insulin": insulin,
        "patient_iob": patient_iob_value,
        "openaps_iob": openaps_iob_value,
    }

    # If attack scenario, add additional information
    if is_attack:
        response["attack_scenario"] = True
        response["real_glucose"] = request.glucose_reading
        response["attack_glucose"] = request.attack_glucose

    return response


@app.get("/patients")
def list_patients():
    """List all active patient simulations"""
    return {"patient_count": len(patient_map), "patients": list(patient_map.keys())}


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    """Remove a patient from the simulation"""
    if patient_id not in patient_map:
        raise HTTPException(status_code=404, detail="Patient not found")

    del patient_map[patient_id]
    return {"status": "deleted", "patient_id": patient_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
