"""
PID Controller Evaluation Utilities.

This module provides functions to evaluate PID controller performance
with multiple error metrics.
"""

import numpy as np


def eval_pid_performance(BG, target_BG, k_P, k_I, k_D, sample_time):
    """
    Evaluate PID controller performance with multiple metrics.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array
        k_P: Proportional gain
        k_I: Integral gain
        k_D: Derivative gain
        sample_time: Sampling time in minutes

    Returns:
        dict: Dictionary containing error metrics (MAX_E, MAE, MSE, RMSE, IAE, ISE)
    """
    print(f"k_P: {k_P}, k_I: {k_I}, k_D: {k_D}, sample_time: {sample_time}")
    target_BG = np.array(target_BG)
    BG = np.array(BG)
    errors = target_BG - BG
    max_e = np.abs(errors).max()
    mae = np.mean(np.abs(errors))  # mean absolute error
    mse = np.mean(errors**2)  # mean square error
    rmse = np.sqrt(mse)  # root mean square error
    iae = np.sum(np.abs(errors) * sample_time)  # integrated absolute error
    ise = np.sum(errors**2 * sample_time)  # integrated square error
    return {
        "MAX_E": max_e,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "IAE": iae,
        "ISE": ise,
    }
