import numpy as np
from .base import Controller, Action
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, target_BG=120, basal_rate=0.2, k_P=0.01, k_I=0.1, k_D=0, sampling_time=5):
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
        
        # State variables
        self.t_last = None
        self.e_integral = 0  # Integral of blood glucose error in mg/dL*min
        self.e_old = None  # Previous blood glucose error for derivative calculation
        self.last_insulin_rate = basal_rate  # Store the last calculated insulin rate
        self.previous_glucose = None  # Store previous glucose for trend analysis
        self.suspend_count = 0  # Count consecutive suspend events
        self.last_normal_dose_time = None  # Last time a normal dose was given
        self.controller_run_count = 0  # Count of controller runs for gradual startup
        
        # Safety parameters
        self.max_insulin_rate = 1.5 * basal_rate  # Reduced maximum insulin rate (1.5 times basal)
        self.max_integral = 300  # Further reduced max integral value
        self.min_integral = -300  # Further reduced min integral value
        self.low_glucose_threshold = 90  # Low glucose threshold (mg/dL)
        self.suspend_threshold = 80  # Suspend threshold (mg/dL)
        self.rapid_drop_threshold = -2  # More sensitive to drops (mg/dL/min)
        self.recovery_threshold = 100  # Threshold for recovery from low glucose (mg/dL)
        self.gradual_recovery_factor = 0.3  # Factor for gradual insulin recovery
        self.prediction_horizon = 30  # Minutes to look ahead for prediction
        
        # Limit rate of insulin increase
        self.max_insulin_increase = 0.05  # Maximum insulin increase per cycle (U/h)
        self.high_glucose_threshold = 180  # High glucose threshold (mg/dL)
        
    def predict_future_glucose(self, current_glucose, rate_of_change, minutes_ahead):
        """Predict future glucose based on current trend"""
        return current_glucose + rate_of_change * minutes_ahead
        
    def policy(self, observation, reward, done, **info):
        # Extract current glucose reading
        current_glucose = observation.CGM
        
        # Get current time - handle case when time is not provided
        if 'time' in info:
            current_time = info['time']
        else:
            # If no time is provided, create a time step based on sampling time
            if self.t_last is None:
                current_time = datetime.now()
            else:
                # Create a new time that's sampling_time minutes after the last time
                current_time = self.t_last + timedelta(minutes=self.sampling_time)
            logger.warning("No 'time' provided in info dictionary. Using simulated time step.")
        
        # Initialize on first call
        if self.t_last is None:
            self.t_last = current_time
            self.e_integral = 0
            self.e_old = None
            self.previous_glucose = current_glucose
            self.last_normal_dose_time = current_time
            self.controller_run_count = 0
            # Start with a conservative dose - 50% of basal if glucose is high
            initial_basal = self.basal_rate * (0.5 if current_glucose > self.target_BG else 0.3)
            logger.info(f"PID Controller initialized - Starting with conservative basal rate: {initial_basal}")
            self.last_insulin_rate = initial_basal
            return Action(basal=initial_basal, bolus=0)
        
        # Increment controller run counter
        self.controller_run_count += 1
        
        # Check if it's time to update (based on sampling time)
        time_diff_minutes = (current_time - self.t_last).total_seconds() / 60 if hasattr(current_time, 'total_seconds') else self.sampling_time
        
        # Calculate glucose rate of change regardless of sampling interval
        glucose_rate_of_change = 0
        if self.previous_glucose is not None and time_diff_minutes > 0:
            glucose_rate_of_change = (current_glucose - self.previous_glucose) / time_diff_minutes
            logger.info(f"Glucose rate of change: {glucose_rate_of_change:.2f} mg/dL/min")
        
        # Predict future glucose for preventive action
        predicted_glucose = self.predict_future_glucose(current_glucose, glucose_rate_of_change, self.prediction_horizon)
        logger.info(f"Predicted glucose in {self.prediction_horizon} min: {predicted_glucose:.1f} mg/dL")
        
        # Only update at sampling intervals
        if time_diff_minutes >= self.sampling_time:
            # Calculate control error (target - actual)
            error = self.target_BG - current_glucose
            logger.info(f"Control error: {error} mg/dL (Target: {self.target_BG} - Current: {current_glucose})")
            
            # ENHANCED SAFETY CHECKS
            
            # Safety Check 1: Predictive suspend - if predicted to go low or already low
            if predicted_glucose <= self.suspend_threshold or current_glucose <= self.suspend_threshold:
                self.suspend_count += 1
                suspend_message = f"SAFETY: {'Current' if current_glucose <= self.suspend_threshold else 'Predicted'} glucose below suspend threshold. Suspending insulin."
                logger.warning(suspend_message)
                self.t_last = current_time
                self.previous_glucose = current_glucose
                # Reset integral to prevent windup during suspension
                self.e_integral = 0
                return Action(basal=0, bolus=0)
            
            # Safety Check 2: If coming out of suspension, use very gradual insulin recovery
            if self.suspend_count > 0:
                # Only gradually restore insulin if glucose is rising and above recovery threshold
                if current_glucose > self.recovery_threshold and glucose_rate_of_change > 0:
                    recovery_rate = self.basal_rate * self.gradual_recovery_factor * min(1.0, current_glucose / self.target_BG)
                    logger.info(f"Gradually resuming insulin at {recovery_rate:.2f} U/h after suspension")
                    self.suspend_count = max(0, self.suspend_count - 1)  # Decrement counter
                    self.t_last = current_time
                    self.previous_glucose = current_glucose
                    self.last_insulin_rate = recovery_rate
                    return Action(basal=recovery_rate, bolus=0)
                else:
                    # Continue suspension if not recovering well enough
                    logger.warning(f"Continuing insulin suspension: glucose={current_glucose}, trend={glucose_rate_of_change:.2f}")
                    self.t_last = current_time
                    self.previous_glucose = current_glucose
                    return Action(basal=0, bolus=0)
            
            # Safety Check 3: Reduce insulin for low glucose or dropping glucose
            if current_glucose < self.low_glucose_threshold or predicted_glucose < self.low_glucose_threshold or glucose_rate_of_change < self.rapid_drop_threshold:
                logger.warning(f"SAFETY: Glucose {current_glucose} mg/dL is low or predicted low or dropping rapidly. Reducing insulin.")
                self.t_last = current_time
                self.previous_glucose = current_glucose
                
                # More aggressive reduction based on how far below threshold and rate of drop
                reduction_factor = 0.4  # More conservative base reduction
                
                # Further reduce if glucose is very low
                if current_glucose < 0.9 * self.low_glucose_threshold:
                    reduction_factor *= 0.6
                
                # Further reduce if dropping rapidly
                if glucose_rate_of_change < 2 * self.rapid_drop_threshold:
                    reduction_factor *= 0.6
                
                reduced_rate = max(0, self.basal_rate * reduction_factor)
                self.last_insulin_rate = reduced_rate
                return Action(basal=reduced_rate, bolus=0)
            
            # Reset suspension count if we've reached normal glucose
            if current_glucose >= self.recovery_threshold and self.suspend_count > 0:
                self.suspend_count = 0
            
            # Anti-windup: Only update integral term when not saturated or when it would decrease saturation
            should_update_integral = (
                (self.e_integral < self.max_integral and error > 0) or
                (self.e_integral > self.min_integral and error < 0) or
                (error > 0 and self.e_integral < 0) or
                (error < 0 and self.e_integral > 0)
            )
            
            # More conservative integral in early runs and with high glucose
            should_update_integral = should_update_integral and (self.controller_run_count > 5 or current_glucose < self.target_BG)
            
            if should_update_integral:
                # Update integral term (convert to hours for consistency with k_I)
                # Use reduced integral gain during first 10 cycles
                effective_k_I = self.k_I
                if self.controller_run_count < 10:
                    effective_k_I = self.k_I * (0.3 + 0.07 * self.controller_run_count)
                    logger.info(f"Using reduced integral gain: {effective_k_I:.4f} (cycle {self.controller_run_count})")
                
                self.e_integral += error / 60 * time_diff_minutes
                # Apply integral limits to prevent windup
                self.e_integral = max(self.min_integral, min(self.max_integral, self.e_integral))
            
            # If glucose is below target, reduce impact of integral term
            if current_glucose < self.target_BG:
                integral_scaling = max(0.2, current_glucose / self.target_BG)  # Scale down as glucose drops
                effective_integral = self.e_integral * integral_scaling
            else:
                # Also reduce integral impact for high glucose to avoid aggressive initial response
                integral_scaling = min(1.0, 2.0 / (1.0 + (current_glucose - self.target_BG) / 100))
                effective_integral = self.e_integral * integral_scaling
            
            # Start with basal rate - using reduced basal during initial cycles
            effective_basal = self.basal_rate
            if self.controller_run_count < 10:
                startup_factor = 0.5 + 0.05 * self.controller_run_count  # Gradually increase from 50% to 100%
                effective_basal = self.basal_rate * startup_factor
                logger.info(f"Startup phase: using {startup_factor*100:.0f}% of basal rate")
            
            insulin_rate = effective_basal
            
            # Use reduced proportional gain during first cycles or with very high glucose
            effective_k_P = self.k_P
            if self.controller_run_count < 10 or current_glucose > self.high_glucose_threshold:
                effective_k_P = self.k_P * (0.5 + 0.05 * min(10, self.controller_run_count))
                logger.info(f"Using reduced proportional gain: {effective_k_P:.4f}")

            # Subtract proportional component 
            insulin_rate -= (effective_k_P / 100) * error
            
            # Subtract integral component with possible scaling
            insulin_rate -= (self.k_I / 100) * effective_integral
            
            # Only add derivative component if we have a previous error and time difference is positive
            if self.e_old is not None and time_diff_minutes > 0:
                # Rate of change (convert to per hour for consistency with k_D)
                de_dt = (error - self.e_old) * 60 / time_diff_minutes
                
                # Use reduced derivative gain during first cycles
                effective_k_D = self.k_D
                if self.controller_run_count < 10:
                    effective_k_D = self.k_D * (0.5 + 0.05 * self.controller_run_count)
                
                # Subtract differential component
                insulin_rate -= (effective_k_D / 100) * de_dt
                
                # Safety: Additional adjustment if blood glucose is changing rapidly
                if de_dt < -100:  # If error is becoming more negative quickly (glucose rising fast)
                    logger.info("Glucose rising rapidly, adjusting insulin rate")
                elif de_dt > 100:  # If error is becoming more positive quickly (glucose dropping fast)
                    # Reduce insulin to prevent low glucose
                    insulin_rate = max(0, insulin_rate * 0.5)  # Even more aggressive reduction
                    logger.info("Glucose dropping rapidly, reducing insulin rate")
            
            # Store current error for next iteration
            self.e_old = error
            
            # Update last time and previous glucose
            self.t_last = current_time
            self.previous_glucose = current_glucose
            
            # Apply safety limits to insulin rate
            insulin_rate = max(0, min(self.max_insulin_rate, insulin_rate))
            
            # Limit rate of increase from previous insulin rate
            if insulin_rate > self.last_insulin_rate:
                # More conservative increase limit during initial cycles
                max_increase = self.max_insulin_increase
                if self.controller_run_count < 10:
                    max_increase = self.max_insulin_increase * (0.3 + 0.07 * self.controller_run_count)
                
                max_new_rate = self.last_insulin_rate + max_increase
                if insulin_rate > max_new_rate:
                    logger.info(f"Limiting insulin increase: {self.last_insulin_rate:.2f} -> {max_new_rate:.2f} U/h (calculated: {insulin_rate:.2f})")
                    insulin_rate = max_new_rate
            
            # Additional safety checks based on current and predicted glucose
            
            # If glucose is trending down and below target, reduce insulin further
            if current_glucose < self.target_BG and glucose_rate_of_change < -1:
                # More aggressive reduction for lower glucose
                reduction_factor = 0.8 - 0.2 * ((self.target_BG - current_glucose) / self.target_BG)
                reduction_factor = max(0.3, reduction_factor)  # Ensure reduction factor is not too extreme
                insulin_rate = max(0, insulin_rate * reduction_factor)
                logger.info(f"Glucose below target and decreasing, reducing insulin by factor {reduction_factor:.2f}")
            
            # If we just increased insulin and glucose is near target, be more conservative
            if current_glucose > 0.9 * self.target_BG and current_glucose < 1.1 * self.target_BG:
                if insulin_rate > 1.2 * self.basal_rate:
                    insulin_rate = self.basal_rate + 0.2 * (insulin_rate - self.basal_rate)
                    logger.info("Near target glucose, moderating insulin increase")
            
            # If glucose is rising from low, be careful with insulin increases
            if glucose_rate_of_change > 1 and current_glucose < self.low_glucose_threshold:
                insulin_rate = min(insulin_rate, self.basal_rate * 0.7)  # More conservative
                logger.info("Rising from low glucose, limiting insulin")
            
            # Special handling for high glucose
            if current_glucose > self.high_glucose_threshold:
                # Ensure we don't overreact to high glucose - cap the insulin
                high_glucose_max = self.basal_rate * (1.0 + (current_glucose - self.high_glucose_threshold) / 200)
                high_glucose_max = min(high_glucose_max, self.max_insulin_rate)
                insulin_rate = min(insulin_rate, high_glucose_max)
                logger.info(f"High glucose handling: limiting insulin to {insulin_rate:.2f} U/h")
            
            # Skip tiny doses
            if insulin_rate < 0.1:
                insulin_rate = 0
            
            basal = insulin_rate
            
            # Remember when we last gave a normal dose
            if basal > 0:
                self.last_normal_dose_time = current_time
            
            # Store the last calculated insulin rate
            self.last_insulin_rate = basal
            
            logger.info(f"PID output - Basal: {basal} U/h")
            
            return Action(basal=basal, bolus=0)
        else:
            # Return previous insulin rate if not at a sampling interval
            # Also check if we need to apply emergency suspension based on new glucose reading
            if current_glucose <= self.suspend_threshold:
                logger.warning(f"SAFETY: Out-of-cycle check - Glucose below suspend threshold. Suspending insulin.")
                self.previous_glucose = current_glucose
                self.suspend_count += 1
                return Action(basal=0, bolus=0)
            
            # Update previous glucose even when not at a sampling interval
            self.previous_glucose = current_glucose
            return Action(basal=self.last_insulin_rate, bolus=0)

    def reset(self):
        """Reset the controller state"""
        self.t_last = None
        self.e_integral = 0
        self.e_old = None
        self.last_insulin_rate = self.basal_rate
        self.previous_glucose = None
        self.suspend_count = 0
        self.last_normal_dose_time = None
        self.controller_run_count = 0
