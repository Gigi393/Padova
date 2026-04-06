"""
Statistical metrics for glucose control evaluation.

This module provides functions to calculate various statistical metrics
for evaluating glucose control performance.
"""

import numpy as np


def get_rmse(BG, target_BG):
    """
    Calculate Root Mean Square Error between BG values and target.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array

    Returns:
        float: RMSE value
    """
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    return np.sqrt(np.mean((BG - target_BG) ** 2))


def get_mae(BG, target_BG):
    """
    Calculate Mean Absolute Error between BG values and target.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array

    Returns:
        float: MAE value
    """
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    errors = target_BG - BG
    return np.mean(np.abs(errors))


def get_mse(BG, target_BG):
    """
    Calculate Mean Square Error between BG values and target.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array

    Returns:
        float: MSE value
    """
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    errors = target_BG - BG
    return np.mean(errors**2)


def get_max_error(BG, target_BG):
    """
    Calculate maximum absolute error between BG values and target.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array

    Returns:
        float: Maximum absolute error
    """
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    errors = target_BG - BG
    return np.abs(errors).max()


def get_iae(BG, target_BG, sample_time):
    """
    Calculate Integrated Absolute Error.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array
        sample_time: Sampling time in minutes

    Returns:
        float: IAE value
    """
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    errors = target_BG - BG
    return np.sum(np.abs(errors) * sample_time)


def get_ise(BG, target_BG, sample_time):
    """
    Calculate Integrated Square Error.

    Args:
        BG: Blood glucose array
        target_BG: Target blood glucose value or array
        sample_time: Sampling time in minutes

    Returns:
        float: ISE value
    """
    BG = np.array(BG)
    target_BG = np.array(target_BG)
    errors = target_BG - BG
    return np.sum(errors**2 * sample_time)
