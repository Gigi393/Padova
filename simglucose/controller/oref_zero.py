import zoneinfo
import logging

import requests
from collections import namedtuple
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from simglucose.controller.base import Controller, Action


logger = logging.getLogger(__name__)

# Named tuple for controller observation
CtrlObservation = namedtuple("CtrlObservation", ["CGM", "bolus"])


class ORefZeroController(Controller):
    """
    OpenAPS oref0 controller that communicates with Node.js server
    for insulin dosage recommendations (single patient per instance)
    """

    DEFAULT_TIMEZONE = "UTC"
    MINIMAL_TIMESTEP = 5  # in minutes

    def __init__(
        self,
        patient_name: str,
        server_url: str = "http://localhost:3000",
        timeout: int = 30,
        profile: Optional[Dict] = None,
    ):
        """
        Initialize the ORefZero controller for a single patient

        Args:
            patient_name: Unique patient identifier
            server_url: URL of the Node.js OpenAPS server
            timeout: Request timeout in seconds
            profile: Patient-specific profile parameters (optional, will use defaults if not provided)
        """
        self.patient_name = self._sanitize_patient_name(patient_name)
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Patient state tracking (single patient)
        self.last_glucose_time = None  # last glucose timestamp
        self.glucose_history = []  # list of glucose readings
        self.meal_history = []  # list of meal entries
        self.bolus_history = []  # list of bolus entries
        self.pump_history = []  # list of pump events
        self.collect_meal = 0  # accumulated meal amount
        self.pending_bolus_entries = (
            []
        )  # list of bolus entries waiting to be sent to oref0
        self.last_insulin = {
            "basal": 0.0,
            "bolus": 0.0,
        }  # last insulin recommendation (from oref0)
        self.last_iob = {}  # last IOB (Insulin on Board) data
        self.last_policy_context = {}  # full context from last calculation
        self.is_initialized = False  # track initialization status

        # Store the required profile
        # Set default profile, but override defaults with any keys present in 'profile'\
        # refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
        self.default_profile = {
            "current_basal": None,  # Current basal rate in U/h
            "sens": 45,  # Insulin Sensitivity Factor (ISF)
            "dia": 7.0,  # Duration of Insulin Action in hours
            "carb_ratio": 10,  # Carb Ratio (g/U)
            "max_iob": 12,  # Maximum insulin on board allowed， # from paper, max 30, from https://androidaps.readthedocs.io/en/latest/DailyLifeWithAaps/KeyAapsFeatures.html
            "max_basal": 4,  # Maximum temporary basal rate in U/h # from paper, max 10
            "max_daily_basal": 0.9,  # Maximum daily basal rate in units per day # from paper
            "max_bg": 140,  # Upper target
            "min_bg": 90,  # Lower target
            "maxCOB": 120,  # Maximum carbs on board  # from oref0 code
            "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 45}]},
            "min_5m_carbimpact": 8,  # Minimum carb absorption rate # from paper and oref0 code
            "max_daily_safety_multiplier": 3,  # Safety multiplier vs max_daily_basal (oref0 default: 3)
            "current_basal_safety_multiplier": 4,  # Safety multiplier vs current basal (oref0 default: 4)
            "autosens_max": 1.2,  # Max autosens ratio (oref0 default: 1.2)
            "autosens_min": 0.7,  # Min autosens ratio (oref0 default: 0.7)
            "type": "current",  # Profile type
        }

        # Build patient profile
        self.patient_profile = self.default_profile.copy()
        if profile:
            self.patient_profile.update(profile)

        logger.info(f"ORefZero Controller initialized for patient: {self.patient_name}")

    @property
    def target_bg(self) -> float:
        """Get the target blood glucose level."""
        return (self.patient_profile["min_bg"] + self.patient_profile["max_bg"]) / 2

    def _sanitize_patient_name(self, patient_name: str) -> str:
        """
        Sanitize patient name by removing special characters that may cause issues.

        Args:
            patient_name: Raw patient name

        Returns:
            Sanitized patient name safe for use as identifier
        """
        return patient_name.replace("#", "")

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to the server with error handling"""
        url = f"{self.server_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == "PATCH":
                response = self.session.patch(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            raise Exception("Server request timed out")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to {url}")
            raise Exception("Could not connect to OpenAPS server")
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error {e.response.status_code} for {url}: {e.response.text}"
            )
            raise Exception(f"HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
            raise Exception("Unexpected error during server request")

    def initialize(self) -> bool:
        """Initialize patient on the server"""
        if self.is_initialized:
            logger.debug(f"Patient {self.patient_name} already initialized")
            return True

        if not self._check_and_correct_profile(self.patient_profile):
            logger.error(f"Invalid profile for patient {self.patient_name}")
            return False

        # Prepare initialization data
        init_data = {
            "profile": self.patient_profile,
            "initialData": {"glucoseHistory": [], "pumpHistory": [], "carbHistory": []},
            "settings": {
                "timezone": self.DEFAULT_TIMEZONE,
                "historyRetentionHours": 24,
                "autoCleanup": False,
            },
        }

        try:
            response = self._make_request(
                "POST", f"/patients/{self.patient_name}/initialize", init_data
            )
            self.is_initialized = True
            logger.info(f"Patient {self.patient_name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize patient {self.patient_name}: {str(e)}")
            return False

    def _check_and_correct_profile(self, profile: Dict):
        # make sure current_basal is set
        if profile["current_basal"] is None:
            logger.error("Profile must include current_basal rate")
            return False

        # make sure min_bg < max_bg
        if profile["min_bg"] > profile["max_bg"]:
            logger.error("Profile min_bg must be less than max_bg")
            return False

        # Auto-set isfProfile from sens (same sensitivity throughout the day)
        profile["isfProfile"] = {
            "sensitivities": [{"offset": 0, "sensitivity": profile["sens"]}]
        }

        # Auto-set max_daily_basal from current_basal (simglucose uses flat basal schedules)
        # max_daily_basal should be the highest basal rate from the patient's schedule
        profile["max_daily_basal"] = profile["current_basal"]

        return True

    @staticmethod
    def _local_timezone() -> datetime:
        return datetime.now().astimezone()

    def _convert_to_utc(self, local_time: datetime) -> datetime:
        if local_time.tzinfo is None:
            # If naive datetime, assume local timezone
            local_time = local_time.replace(tzinfo=self._local_timezone().tzinfo)
        utc_time = local_time.astimezone(zoneinfo.ZoneInfo(self.DEFAULT_TIMEZONE))
        # Remove the timezone info (make it naive)
        return utc_time.replace(tzinfo=None)

    def _convert_time_to_timestamp(self, time_obj) -> str:
        """Convert time object to ISO timestamp string"""
        if isinstance(time_obj, str):

            # Assume it's already a valid timestamp
            try:
                time_obj = datetime.fromisoformat(time_obj.rstrip("Z"))
            except:
                raise ValueError(
                    f"Invalid timestamp format, could not convert {time_obj} to datetime"
                )
        elif isinstance(time_obj, (int, float)):
            # Assume it's a Unix timestamp
            try:
                time_obj = datetime.fromtimestamp(time_obj)
            except:
                raise ValueError(
                    f"Invalid timestamp format, could not convert {time_obj} to datetime"
                )

        # now we have a datetime object
        if not isinstance(time_obj, datetime):
            raise ValueError(f"Expected datetime object, got {type(time_obj)}")
        # Convert to UTC and format as ISO string
        utc_time = self._convert_to_utc(time_obj)
        return utc_time.isoformat() + "Z"  # Append 'Z' to indicate

    def _prepare_new_data(
        self,
        glucose: float,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Prepare new data to send to the server"""
        new_data = {}

        # Add glucose reading if we have one
        if glucose > 0:
            # Convert UTC timestamp string to local datetime, then to int (epoch seconds)
            utc_dt = datetime.fromisoformat(timestamp.rstrip("Z")).replace(
                tzinfo=zoneinfo.ZoneInfo(self.DEFAULT_TIMEZONE)
            )
            glucose_entry = {
                "date": int(utc_dt.timestamp() * 1000),  # Convert to milliseconds
                "glucose": glucose,
                "timestamp": timestamp,
            }
            new_data["glucoseReadings"] = [glucose_entry]
            self.glucose_history.append(glucose_entry)

        # Add carb entry if we have a meal
        if self.collect_meal > 0:
            carb_entry = {
                "timestamp": timestamp,
                "carbs": self.collect_meal,
            }
            new_data["carbEntries"] = [carb_entry]
            self.meal_history.append(carb_entry)
            self.collect_meal = 0

        # Add all pending bolus entries (with their actual delivery timestamps)
        if len(self.pending_bolus_entries) > 0:
            new_data["bolusEntries"] = self.pending_bolus_entries.copy()
            self.bolus_history.extend(self.pending_bolus_entries)
            self.pending_bolus_entries = (
                []
            )  # Clear pending boluses after sending to oref0

        # Add pump history entries (insulin deliveries from previous actions)
        if len(self.pump_history) > 0:
            new_data["pumpHistory"] = self.pump_history.copy()
            # Clear the pump history after sending
            self.pump_history = []

        return new_data

    def policy(
        self,
        observation,
        reward: float,
        done: bool,
        meal: float,
        time,
    ) -> Action:
        """
        Get insulin dosage recommendation from OpenAPS server

        Args:
            observation: CtrlObservation object with glucose level
            reward: Reward signal (not used by OpenAPS)
            done: Episode done flag (not used by OpenAPS)
            meal: Carbohydrate amount in grams
            time: Current simulation time

        Returns:
            CtrlAction with basal and bolus insulin recommendations
        """
        # Initialize patient if not already done
        if not self.is_initialized:
            if not self.initialize():
                logger.warning(f"Failed to initialize patient {self.patient_name}")
                raise ValueError("Failed to initialize patient.")

        # Convert time to timestamp
        timestamp = self._convert_time_to_timestamp(time)
        previous_timestamp = self.last_glucose_time

        # Accumulate meal data
        self.collect_meal += meal

        # Record meal bolus with its actual delivery timestamp (if any)
        # This is for oref0 IOB tracking only, not for returning to patient
        if observation.bolus > 0:
            bolus_entry = {
                "timestamp": timestamp,
                "bolus": observation.bolus,
            }
            self.pending_bolus_entries.append(bolus_entry)

        # If previous_timestamp is not None and timestamp difference < MINIMAL_TIMESTEP
        # Return early without updating oref0
        if previous_timestamp is not None and (
            datetime.fromisoformat(timestamp.rstrip("Z"))
            < datetime.fromisoformat(previous_timestamp.rstrip("Z"))
            + timedelta(minutes=self.MINIMAL_TIMESTEP)
        ):
            # Return last oref0 recommendation (basal and oref0's bolus only)
            # Meal bolus will be added by the wrapper controller
            return Action(
                basal=self.last_insulin["basal"],
                bolus=self.last_insulin["bolus"],  # oref0's bolus (SMB/microbolus)
            )

        # Extract glucose level
        glucose_level = observation.CGM

        # Prepare new data (will include all pending bolus entries with their timestamps)
        new_data = self._prepare_new_data(glucose_level, timestamp)
        print(new_data)
        # Prepare calculation request
        calc_data = {
            "currentTime": timestamp,
            "newData": new_data,
            "options": {
                "microbolus": False,  # Disable, since oref0 doesn't support microbolus, though it might be available in the future for oref1
                "overrideProfile": {},  # Could be used for temporary profile changes
            },
        }

        # Make calculation request
        endpoint = f"/patients/{self.patient_name}/calculate"
        response = self._make_request("POST", endpoint, calc_data)
        print(response["suggestion"])
        # Extract recommendation
        suggestion = response.get("suggestion", {})
        basal_rate = response.get("IIR", 0.0) / 60  # U/h -> U/min
        iob_data = response.get("context", {}).get("iob", {})

        # Calculate bolus recommendation
        # OpenAPS typically provides basal adjustments, bolus calculation
        # might need to be derived from the suggestion
        bolus_amount = 0.0

        # Check if there's a microbolus recommendation
        if "units" in suggestion:
            bolus_amount = suggestion.get("units", 0.0)

        # Check for SMB (Super Micro Bolus) recommendation
        if "microbolus" in suggestion:
            bolus_amount += suggestion.get("microbolus", 0.0)

        # Extract IOB value for logging (iob_data contains the full IOB object)
        iob_value = iob_data.get("iob", 0.0) if iob_data else 0.0
        max_iob = self.patient_profile["max_iob"]

        # Log the recommendation
        logger.debug(
            f"Patient {self.patient_name}: BG={glucose_level:.1f}, "
            f"Meal={meal:.1f}g, IOB={iob_value:.2f}U (max={max_iob:.1f}), "
            f"Basal={basal_rate:.3f}, Bolus={bolus_amount:.3f}"
        )

        # Store the last glucose time, insulin recommendation, and IOB
        self.last_glucose_time = timestamp
        self.last_insulin = {"basal": basal_rate, "bolus": bolus_amount}
        self.last_iob = iob_data

        # Store policy context with IOB-related values
        self.last_policy_context = {
            "iob_value": iob_value,  # OpenAPS calculated IOB
            "max_iob": max_iob,  # Safety limit from profile
        }

        return Action(basal=basal_rate, bolus=bolus_amount)

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current patient status from the server"""
        try:
            if not self.is_initialized:
                logger.warning(f"Patient {self.patient_name} not initialized")
                return None

            response = self._make_request("GET", f"/patients/{self.patient_name}/status")
            return response

        except Exception as e:
            logger.error(f"Error getting patient status for {self.patient_name}: {str(e)}")
            return None

    def update_profile(self, profile_updates: Dict[str, Any]) -> bool:
        """Update patient profile on the server"""
        try:
            if not self.is_initialized:
                logger.warning(f"Patient {self.patient_name} not initialized")
                return False

            response = self._make_request(
                "PATCH", f"/patients/{self.patient_name}/profile", profile_updates
            )

            # Update local copy
            self.patient_profile.update(profile_updates)

            logger.info(
                f"Updated profile for patient {self.patient_name}: {list(profile_updates.keys())}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating patient profile for {self.patient_name}: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if the OpenAPS server is responding"""
        try:
            response = self._make_request("GET", "/health")
            logger.info(f"Server health check: {response.get('message', 'OK')}")
            return True
        except Exception as e:
            logger.error(f"Server health check failed: {str(e)}")
            return False

    def get_iob(self) -> Optional[Dict[str, Any]]:
        """
        Get IOB-related values from OpenAPS calculation

        Returns:
            Dict with keys:
                - iob_value: OpenAPS calculated total IOB (float)
                - max_iob: Maximum allowed IOB from profile (float)

        Note:
            To get the patient model's physiological IOB, use patient.get_iob()
            method on the T1DMPatient instance in your simulation loop.
        """
        return self.last_policy_context if self.last_policy_context else None

    def get_policy_context(self) -> Optional[Dict[str, Any]]:
        """
        Get the full policy context from last calculation

        Returns same as get_iob (they return the same data)
        """
        return self.get_iob()

    def get_profile(self) -> Dict[str, Any]:
        """Get the patient profile"""
        return self.patient_profile
