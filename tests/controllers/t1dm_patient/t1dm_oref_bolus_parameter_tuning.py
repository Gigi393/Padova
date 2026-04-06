"""
T1DM OpenAPS Parameter Tuning using CREATE Trial Settings.

This module implements parameter tuning for Type 1 Diabetes patients using the
AnyDANA-Loop (OpenAPS) configuration from the CREATE Trial.

CREATE Trial Reference:
    Title: "Closed-Loop Insulin Delivery in Children with Type 1 Diabetes"
    Journal: New England Journal of Medicine (2022)
    DOI: https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    Supplementary Appendix: Table S2 - Technical Manual for AnyDANA-Loop

Parameters Being Tuned:
    - sens: Insulin Sensitivity Factor (ISF) in mg/dL per unit
    - dia: Duration of Insulin Action in hours
    - max_iob: Maximum Insulin On Board in units
    - min_bg: Lower glucose target in mg/dL
    - max_bg: Upper glucose target in mg/dL

Tuning Ranges (CREATE Trial):
    Children (<16 years):
        - sens: 50-100 mg/dL/U
        - dia: 5-8 hours
        - max_iob: 15-20u
        - min_bg: 70-90 mg/dL (3.9-5.0 mmol/L)
        - max_bg: 117-180 mg/dL (6.5-10.0 mmol/L)

    Adults (>18 years):
        - sens: 30-50 mg/dL/U
        - dia: 5-8 hours
        - max_iob: 25-30u
        - min_bg: 70-90 mg/dL (3.9-5.0 mmol/L)
        - max_bg: 117-180 mg/dL (6.5-10.0 mmol/L)

Fixed Parameters (in profiles):
    - max_basal: 4 U/hr
    - carb_ratio: Patient-specific (from simglucose model)
    - current_basal: Patient-specific (from simglucose model)

Controller Default Parameters (oref_zero.py):
    - max_daily_safety_multiplier: 3
    - current_basal_safety_multiplier: 4
    - autosens_max: 1.2
    - autosens_min: 0.7
    - maxCOB: 120g
    - min_5m_carbimpact: 8 mg/dL/5min

Auto-Configured Parameters (oref_zero.py):
    - isfProfile: Auto-set from sens
    - max_daily_basal: Auto-set from current_basal
"""

import logging
import matplotlib
from pathlib import Path
from collections import namedtuple
from typing import Any, Dict, List, Optional
import numpy as np
from itertools import product

from simglucose.patient.t1dm_patient import T1DMPatient, Action
from simglucose.controller.oref_zero_with_meal_bolus import ORefZeroWithMealBolus, CtrlObservation
from analytics import PatientType
from tuning import OpenAPSParameterTuningBase

matplotlib.use("Agg")  # Use non-interactive backend to prevent window pop-ups

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent
result_dir = parent_folder / "imgs" / "oref0_parameter_tuning"


class T1DMOpenAPSParameterTuning(OpenAPSParameterTuningBase):
    """
    T1DM-specific OpenAPS parameter tuning implementation.

    This class extends OpenAPSParameterTuningBase to provide T1DM patient-specific
    parameter tuning using the ORef0 controller.
    """

    def set_patients_by_group(self, patients: Dict[PatientType, List[str]]):
        """
        Set the patient groups for parameter tuning.

        Args:
            patients: Dictionary mapping PatientType to list of patient names
        """
        self._patients_by_group = patients

    @staticmethod
    def parse_virtual_patient_id(virtual_patient_id: str) -> tuple[str, int, int]:
        """
        Parse a virtual patient ID to extract patient name, parameter index, and carb amount.

        Args:
            virtual_patient_id: Virtual patient ID (e.g., "adult_001_param_0_carb_50")

        Returns:
            Tuple of (patient_name, param_idx, carb_amount)
            e.g., ("adult#001", 0, 50)
        """
        parts = virtual_patient_id.split("_")

        # Find the index of "param"
        param_idx_pos = parts.index("param")

        # Patient name is everything before "param"
        patient_parts = parts[:param_idx_pos]
        patient_name = "#".join(patient_parts)  # Restore # separator

        # Parameter index is right after "param"
        param_idx = int(parts[param_idx_pos + 1])

        # Carb amount is after "carb"
        carb_idx_pos = parts.index("carb")
        carb_amount = int(parts[carb_idx_pos + 1])

        return patient_name, param_idx, carb_amount

    def create_virtual_patient_id(
        self, patient_name: str, param_idx: int, carb_amount: int, trial: Optional[int] = None
    ) -> str:
        """
        Create a virtual patient ID that encodes patient, parameter index, carb amount, and trial.

        Args:
            patient_name: Patient name (e.g., "adult#001")
            param_idx: Parameter set index
            carb_amount: Amount of carbs in grams (e.g., 50)
            trial: Trial number for repeated runs (e.g., 0, 1, 2)

        Returns:
            Virtual patient ID (e.g., "adult_001_param_0_carb_50_trial_0")
        """
        base_name = patient_name.replace("#", "_")
        if trial is not None:
            return f"{base_name}_param_{param_idx}_carb_{carb_amount}_trial_{trial}"
        return f"{base_name}_param_{param_idx}_carb_{carb_amount}"

    @staticmethod
    def simulation_loop(virtual_patient_id, patient_map):
        """
        Simulation loop for T1DM patient with ORef0 + Meal Bolus controller.

        This implements the closed-loop control simulation for Type 1 Diabetes patients
        using the OpenAPS ORef0 algorithm with meal announcement bolus.

        Args:
            virtual_patient_id: Virtual patient ID
            patient_map: Dictionary mapping virtual patient IDs to (patient_name, profile, carb_amount)

        Returns:
            Tuple of (t, BG, CHO, insulin, target_bg)
        """
        patient_name, _, carb_amount = T1DMOpenAPSParameterTuning.parse_virtual_patient_id(virtual_patient_id)

        p = T1DMPatient.withName(patient_name)

        profile = patient_map[virtual_patient_id][1]

        if profile is not None:
            profile["carb_ratio"] = p.carb_ratio
            profile["current_basal"] = p.basal * 60  # U/min to U/h

        # Fixed simulation time and meal delivery at 20 minutes
        max_simulation_time = 1450  # 24 hours + 10 minutes
        meal_time = 20  # Deliver meal at 20 minutes

        # Create meal schedule - single meal at meal_time
        meal_schedule = [(meal_time, carb_amount)]

        ctrl = ORefZeroWithMealBolus(
            patient_name=virtual_patient_id,
            server_url="http://localhost:3000",
            timeout=30000,  # TODO: remove timeout when not in debug
            profile=profile,
            meal_schedule=meal_schedule,
            carb_factor=(
                profile["carb_ratio"] if profile and "carb_ratio" in profile else 10
            ),
            release_time_before_meal=10,  # Release bolus 10 minutes before meal
            carb_estimation_error=0.3,  # 30% carb estimation error
            t_start=p.t_start,
        )

        if not ctrl.initialize():
            logger.error(
                f"Failed to initialize controller for patient {virtual_patient_id}"
            )
            return None

        t = []
        CHO = []
        insulin = []
        BG = []

        while p.t_elapsed < max_simulation_time:
            # Deliver meal at exactly 20 minutes
            carb = carb_amount if int(p.t_elapsed) == meal_time else 0

            ctrl_obs = CtrlObservation(CGM=p.observation.Gsub)

            if p.observation.Gsub < 39:
                print("Patient is dead")
                break

            ctrl_action = ctrl.policy(
                observation=ctrl_obs,
                reward=0,
                done=False,
                meal=carb,
                time=p.t,
            )

            ins = ctrl_action.basal + ctrl_action.bolus
            act = Action(insulin=ins, CHO=carb)  # U/min
            p.step(act)

            t.append(p.t_elapsed)
            CHO.append(act.CHO)
            insulin.append(act.insulin)
            BG.append(p.observation.Gsub)
            print(
                f"\033[94mt: {p.t_elapsed}, t: {p.t} BG: {p.observation.Gsub}, CHO: {carb}, Insulin: {ins}\033[0m"
            )

        return t, BG, CHO, insulin, ctrl.target_bg

    def generate_profiles_by_group(self, group: PatientType) -> List[Dict[str, Any]]:
        """
        Generate OpenAPS parameter profiles for T1DM patient groups.

        This method creates parameter combinations based on literature values
        for different age groups, including tunable glucose target ranges.

        Args:
            group: Patient type group (CHILD or ADULT)

        Returns:
            List of profile dictionaries with OpenAPS parameters

        Note:
            Glucose target ranges:
            - Algorithm control target (CREATE Trial): 90-117 mg/dL (5.0-6.5 mmol/L)
            - Clinical outcome target (CREATE Trial): 70-180 mg/dL (3.9-10.0 mmol/L)
            Both min_bg and max_bg are now tunable parameters in parameter_group.
        """
        max_basal = 4

        # Define default profiles for each age group based on CREATE Trial and oref0 defaults
        # Variable names match oref0 repository: D:\Repos\SimglucoseProjects\oref0\lib\profile\index.js
        patient_group_default_profiles = {
            PatientType.CHILD: {
                # PROFILE SETTINGS
                "sens": 150,  # ISF (oref0: lib/profile/index.js:169)
                "dia": 7,  # Duration of Insulin Action in hours (oref0: lib/profile/index.js:119)
                "carb_ratio": 30,  # I:C ratio (oref0: lib/profile/index.js:176)
                "min_bg": 90,  # Lower glucose target
                "max_bg": 110,  # Upper glucose target
                # "isfProfile": {
                #     "sensitivities": [{"offset": 0, "sensitivity": 150}] # same as sens
                # },
                # SAFETY SETTINGS
                "max_iob": 3,
                "max_basal": max_basal,
            },
            PatientType.ADULT: {
                # PROFILE SETTINGS
                "sens": 45,  # ISF (oref0: lib/profile/index.js:169)
                "dia": 7.0,  # Duration of Insulin Action in hours (oref0: lib/profile/index.js:119)
                "carb_ratio": 20,  # I:C ratio (oref0: lib/profile/index.js:176)
                "min_bg": 90,  # Lower glucose target
                "max_bg": 110,  # Upper glucose target
                # "isfProfile": {
                #     "sensitivities": [{"offset": 0, "sensitivity": 60}] # same as sens
                # },
                # SAFETY SETTINGS
                "max_iob": 12,
                "max_basal": max_basal,
            },
        }

        # Parameter ranges based on CREATE Trial Supplementary Appendix
        # Variable names match oref0 repository: D:\Repos\SimglucoseProjects\oref0\lib\profile\index.js
        parameter_group = {
            PatientType.CHILD: {
                # PROFILE SETTINGS (oref0: lib/profile/index.js)
                "sens": {
                    "step_count": 5,
                    "range": (50, 150),
                },  # ISF 1:50 to 1:150 mg/dL (CREATE: GPT recommendation)
                "dia": {
                    "step_count": 3,
                    "range": (5, 8),
                },  # DIA 5-8 hours (CREATE: Table S2, Page 9)
                "min_bg": {
                    "step_count": 3,
                    "range": (70, 90),
                },  # Lower glucose target: 70-90 mg/dL (3.9-5.0 mmol/L)
                "max_bg": {
                    "step_count": 4,
                    "range": (117, 180),
                },  # Upper glucose target: 117-180 mg/dL (6.5-10.0 mmol/L)
                # SAFETY SETTINGS (oref0: lib/profile/index.js:15-19)
                "max_iob": {
                    "step_count": 3,
                    "range": (10, 20),
                },  # Max IOB 15-20u for children (CREATE: Table S2, Page 14, Age <16)
            },
            PatientType.ADULT: {
                # PROFILE SETTINGS (oref0: lib/profile/index.js)
                "sens": {
                    "step_count": 4,
                    "range": (30, 50),
                },  # ISF 1:30 to 1:50 mg/dL (CREATE: GPT recommendation)
                "dia": {
                    "step_count": 3,
                    "range": (5, 8),
                },  # DIA 5-8 hours (CREATE: Table S2, Page 9)
                "min_bg": {
                    "step_count": 3,
                    "range": (70, 90),
                },  # Lower glucose target: 70-90 mg/dL (3.9-5.0 mmol/L)
                "max_bg": {
                    "step_count": 4,
                    "range": (117, 180),
                },  # Upper glucose target: 117-180 mg/dL (6.5-10.0 mmol/L)
                # SAFETY SETTINGS (oref0: lib/profile/index.js:15-19)
                "max_iob": {
                    "step_count": 4,
                    "range": (10, 30),
                },  # Max IOB 25-30u for adults (CREATE: Table S2, Page 14, Age >18)
            },
        }

        # Get base profile and parameter ranges for the group
        base_profile = patient_group_default_profiles[group]
        param_ranges = parameter_group[group]

        # Generate value arrays for each parameter
        param_values = {}
        for param_name, param_config in param_ranges.items():
            step_count = param_config["step_count"]
            min_val, max_val = param_config["range"]
            param_values[param_name] = np.linspace(min_val, max_val, step_count)

        # Generate all combinations
        param_names = list(param_values.keys())
        value_combinations = product(*[param_values[name] for name in param_names])

        # Create profiles
        profiles = []
        for values in value_combinations:
            profile = base_profile.copy()

            # Update each parameter value
            sens_value = None
            for param_name, value in zip(param_names, values):
                profile[param_name] = value
                if param_name == "sens":
                    sens_value = value

            # Update isfProfile sensitivity to match sens value
            if sens_value is not None:
                profile["isfProfile"] = {
                    "sensitivities": [{"offset": 0, "sensitivity": sens_value}]
                }

            profiles.append(profile)

        return profiles

    def generate_patient_map_and_simulation_config(
        self,
        run_dir: Path,
    ) -> tuple[Dict[str, tuple[str, Dict[str, Any], int]], List[tuple[str, Path]]]:
        """
        Generate patient map and simulation configs for T1DM OpenAPS parameter tuning.

        Args:
            run_dir: Directory to save simulation results

        Returns:
            Tuple of (patient_map, simulation_configs)
            - patient_map: Dictionary mapping virtual_patient_id -> (patient_name, profile, carb_amount)
            - simulation_configs: List of tuples (virtual_patient_id, img_save_dir)
        """
        # Get experiment configuration
        if self._patients_by_group is None:
            raise ValueError("Patients by group not set for parameter tuning.")

        # Generate 10 carb amounts from normal distribution (15-130g)
        # Using mean of 70g and std of 30g
        np.random.seed(42)  # For reproducibility
        carb_amounts = np.random.normal(loc=70, scale=30, size=10)
        carb_amounts = np.clip(
            carb_amounts, 15, 130
        )  # Ensure values stay within 15-130g
        carb_amounts = sorted(
            [int(round(carb)) for carb in carb_amounts]
        )  # Round to integers and sort

        # Generate profiles for each patient group
        patient_profiles_by_group = {}
        for group in self._patients_by_group.keys():
            patient_profiles_by_group[group] = self.generate_profiles_by_group(group)

        # Build patient map and simulation configs
        patient_map = {}
        simulation_configs = []

        # Number of trials to repeat each configuration (for carb estimation error randomness)
        num_trials = 3

        for group in self._patients_by_group.keys():
            profiles = patient_profiles_by_group[group]

            for patient_name in self._patients_by_group[group]:
                # Create single patient-specific directory
                patient_folder = patient_name.replace("#", "_")
                patient_dir = run_dir / patient_folder
                patient_dir.mkdir(exist_ok=True, parents=True)

                for param_idx, profile in enumerate(profiles):
                    for carb_amount in carb_amounts:
                        # Repeat each configuration num_trials times
                        for trial in range(num_trials):
                            virtual_patient_id = self.create_virtual_patient_id(
                                patient_name, param_idx, carb_amount, trial
                            )
                            patient_map[virtual_patient_id] = (
                                patient_name,
                                profile,
                                carb_amount,
                            )

                            simulation_configs.append(
                                (
                                    virtual_patient_id,
                                    patient_dir,
                                )
                            )

        # DEBUG: Uncomment the next line to test with only first 4 configs
        # simulation_configs = simulation_configs[:4]

        return patient_map, simulation_configs


if __name__ == "__main__":
    """
    T1DM OpenAPS parameter tuning experiment.

    Following paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    Resplit patients into 2 groups: children (7-12), adults (16-70)
    Selected patients:
        child: child#002, child#008, child#010
        adult: adolescent#003, adult#006, adult#009

    Goal: Find optimal parameter set for each patient that achieves
    realistic time-in-range distribution following the paper.
    """
    # Define patient groups for parameter tuning
    patients_by_group = {
        PatientType.CHILD: ["child#002"],
        # PatientType.CHILD: ["child#008"],
        # PatientType.ADULT: ["adolescent#003"],
        PatientType.ADULT: ["adult#007"],
    }

    # Create and run the experiment
    # Limit to 6 parallel workers (good balance for 14-core system)
    experiment = T1DMOpenAPSParameterTuning(output_dir=result_dir, max_workers=6)
    experiment.set_patients_by_group(patients_by_group)
    results = experiment.run()

    logger.info(f"Experiment completed with {len(results)} results")
