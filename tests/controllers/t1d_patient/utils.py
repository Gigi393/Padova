"""
Utility functions for working with T1D patients in tests.
"""

import pandas as pd
from simglucose.patient.t1dpatient import PATIENT_PARA_FILE
from simglucose.patient.t1dm_patient import PatientType


def get_patients():
    """
    Get list of all patient names from patient parameter file.

    Returns:
        list: List of patient names
    """
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    return patient_params.Name.tolist()


def get_patient_by_group(patient_type: PatientType):
    """
    Get list of patient names for a specific patient type/group.

    Args:
        patient_type: PatientType enum (ADOLESCENT, ADULT, or CHILD)

    Returns:
        list: List of patient names for the specified group
    """
    if patient_type == PatientType.ADOLESCENT:
        return [f"adolescent#00{i}" for i in range(1, 10)]
    elif patient_type == PatientType.ADULT:
        return [f"adult#00{i}" for i in range(1, 10)]
    elif patient_type == PatientType.CHILD:
        return [f"child#00{i}" for i in range(1, 10)]
