import logging
from typing import Dict, Any, Optional
from collections import namedtuple
from simglucose.controller.oref_zero import ORefZeroController, CtrlObservation as ORefCtrlObservation
from simglucose.controller.meal_bolus_ctrller import MealAnnouncementBolusController
from simglucose.controller.base import Controller, Action

# Simple observation for external use - only CGM needed
CtrlObservation = namedtuple("CtrlObservation", ["CGM"])

logger = logging.getLogger(__name__)


class ORefZeroWithMealBolus(Controller):
    """
    Combined controller that uses composition to combine:
    - ORefZero for basal rate calculations
    - MealAnnouncementBolusController for predictive meal bolus calculations

    This controller uses composition (has-a relationship) rather than inheritance
    to keep the two controllers separate and avoid variable conflicts.

    This is a single-patient controller - create one instance per patient.
    """

    def __init__(
        self,
        patient_name: str,
        server_url: str = "http://localhost:3000",
        timeout: int = 30,
        profile: Optional[Dict] = None,
        meal_schedule=None,
        carb_factor=10,
        release_time_before_meal=10,
        carb_estimation_error=0.3,
        t_start=None,
    ):
        """
        Initialize the combined controller for a single patient.

        Args:
            patient_name: Unique patient identifier
            server_url: URL of the Node.js OpenAPS server
            timeout: Request timeout in seconds
            profile: Patient-specific ORefZero profile parameters (optional)
            meal_schedule: List of tuples (time_minutes, carbs_grams), e.g.,
                          [(120, 50), (360, 75), (720, 60)]
            carb_factor: Carbohydrate factor in g/U (default: 10, meaning 1U per 10g CHO)
            release_time_before_meal: Time in minutes to release bolus before meal (default: 10)
            carb_estimation_error: Percentage of error in carbohydrate estimation (e.g., 0.3 for +/- 30%)
            sample_time: Time period over which to deliver bolus in minutes (default: 1)
            t_start: Patient simulation start time as datetime object (optional)
        """
        self.patient_name = patient_name

        # Create ORefZeroController instance for this patient
        self.oref0_controller = ORefZeroController(
            patient_name=patient_name,
            server_url=server_url,
            timeout=timeout,
            profile=profile,
        )

        # Create MealAnnouncementBolusController instance for this patient
        self.meal_bolus_controller = MealAnnouncementBolusController(
            meal_schedule=meal_schedule,
            carb_factor=carb_factor,
            release_time_before_meal=release_time_before_meal,
            carb_estimation_error=carb_estimation_error,
            t_start=t_start,
        )

        logger.info(f"ORefZeroWithMealBolus Controller initialized for patient: {patient_name}")

    def policy(
        self,
        observation,
        reward: float,
        done: bool,
        meal: float,
        time,
    ) -> Action:
        """
        Get insulin dosage recommendation combining ORefZero basal with meal bolus.

        Args:
            observation: Simple observation with just CGM field (e.g., SimpleObservation or any object with .CGM attribute)
            reward: Reward signal (not used by OpenAPS)
            done: Episode done flag (not used by OpenAPS)
            meal: Carbohydrate amount in grams (from environment)
            time: Current simulation time (datetime object)

        Returns:
            Action with combined basal (from ORefZero) and bolus (meal announcement + ORefZero)
        """

        # Get meal announcement bolus (if any meal is upcoming)
        # MealAnnouncementBolusController will calculate elapsed_time from time - t_start
        meal_bolus_action = self.meal_bolus_controller.policy(time)

        # Create ORefZero observation with meal bolus (internal use only)
        # This adds the meal bolus to the observation so ORefZero can account for it
        observation_with_bolus = ORefCtrlObservation(
            CGM=observation.CGM, bolus=meal_bolus_action.bolus
        )

        # Get ORefZero recommendation (basal and any bolus from ORefZero)
        # ORefZero expects datetime object
        oref_action = self.oref0_controller.policy(
            observation_with_bolus, reward, done, meal, time
        )

        # Combine: use ORefZero basal, add meal bolus to ORefZero bolus
        combined_bolus = oref_action.bolus + meal_bolus_action.bolus

        logger.debug(
            f"Patient {self.patient_name} - ORefZero basal: {oref_action.basal:.3f}, "
            f"ORefZero bolus: {oref_action.bolus:.3f}, "
            f"Meal bolus: {meal_bolus_action.bolus:.3f}, "
            f"Total bolus: {combined_bolus:.3f}"
        )

        return Action(basal=oref_action.basal, bolus=combined_bolus)

    def initialize(self) -> bool:
        """Initialize patient on ORefZero controller."""
        return self.oref0_controller.initialize()

    def get_iob(self) -> Optional[Dict[str, Any]]:
        """Get IOB-related values from ORefZero controller."""
        return self.oref0_controller.get_iob()

    def get_policy_context(self) -> Optional[Dict[str, Any]]:
        """Get the full policy context from ORefZero controller."""
        return self.oref0_controller.get_policy_context()

    def get_profile(self) -> Dict[str, Any]:
        """Get patient profile from ORefZero controller."""
        return self.oref0_controller.get_profile()

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current patient status from ORefZero controller."""
        return self.oref0_controller.get_status()

    def update_profile(self, profile_updates: Dict[str, Any]) -> bool:
        """Update patient profile on ORefZero controller."""
        return self.oref0_controller.update_profile(profile_updates)

    @property
    def target_bg(self) -> float:
        """Get the target blood glucose level from ORefZero controller."""
        return self.oref0_controller.target_bg
