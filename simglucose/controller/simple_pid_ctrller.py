"""A simple PID controller

@date: 2020.03.26
@editor: Kexin Wei

"""

import numpy as np
from .base import Controller, Action
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class SimplePIDController(Controller):

    def __init__(
        self,
        target_BG=120,
        basal_rate=0.2,
        k_P=0.01,
        k_I=0.1,
        k_D=0,
        sampling_time=5,
    ):
        """
        PID Controller for insulin delivery

        Args:
            target_BG (float): Target blood glucose in mg/dL
            basal_rate (float): Default basal rate in U/h
            k_P (float): Proportional gain in U/h / (100 mg/dL)
            k_I (float): Integral gain in U/h / (100 mg/dL) / h
            k_D (float): Derivative gain in U/h / (100 mg/dL) * h
            sampling_time (int): Sampling time in minutes
        """
        self.target_BG = target_BG
        self.basal_rate = basal_rate
        self.k_P = k_P
        self.k_I = k_I
        self.k_D = k_D
        self.sampling_time = sampling_time

        self.reset()

    def reset(self):
        """Reset the controller state"""
        self.t_last = None
        self.e_integral = 0
        self.e_old = None
        self.last_insulin_rate = self.basal_rate
        self.suspend = 0

    def policy(self, observation, reward, done, **info) -> Action:
        current_glucose = observation.CGM
        if "time" not in info:
            SystemExit("Time not provided in info dictionary")
        current_time = info["time"]

        # Initialize on first call
        if self.t_last is None:
            self.t_last = current_time
            return Action(basal=0, bolus=0)

        time_diff_minutes = current_time - self.t_last
        logger.debug(f"Time since last update: {time_diff_minutes} minutes")

        if time_diff_minutes < self.sampling_time:
            return Action(basal=0, bolus=0)

        error = self.target_BG - current_glucose
        if error >= 0:
            if self.suspend == False:
                logger.warning(f"Glucose too low, insulin suspend")
            self.suspend = True
            return Action(basal=0, bolus=0)

        self.suspend = False
        self.e_integral += error * self.sampling_time

        # insulin_rate = self.basal_rate - self.k_P * error - self.k_I * self.e_integral
        insulin_rate = self.basal_rate - self.k_P * error - self.k_I * self.e_integral

        if self.e_old is not None:
            de_dt = (error - self.e_old) / self.sampling_time
            insulin_rate -= self.k_D * de_dt

        insulin_rate = max(0, insulin_rate)

        self.t_last = current_time
        self.e_old = error

        return Action(basal=insulin_rate, bolus=0)
