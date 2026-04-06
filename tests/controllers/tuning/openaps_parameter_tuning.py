"""
Parallel Simulation Framework for Glucose Control Analytics.

This module provides reusable functions for running parallel simulations
of patient-controller closed loop systems across different repositories.
"""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import multiprocessing

from analytics.time_in_range import TIRConfig
from plotting.plot import plot_and_save_with_tir
from analytics.patient_types import PatientType


logger = logging.getLogger(__name__)


class OpenAPSParameterTuningBase(ABC):
    """
    Base class for OpenAPS parameter tuning experiments.

    This class provides the core framework for running parameter tuning experiments
    across different patient types and OpenAPS controller variants. Subclasses should
    implement the abstract methods to customize behavior for specific use cases.
    """

    def __init__(self, output_dir: Path, tir_config: Optional[TIRConfig] = None, max_workers: Optional[int] = None):
        """
        Initialize the parameter tuning experiment.

        Args:
            output_dir: Base directory for saving results
            tir_config: TIR configuration (defaults to BASIC if not provided)
            max_workers: Maximum number of parallel workers (defaults to CPU count if not provided)
        """
        self.output_dir = output_dir
        self.tir_config = tir_config or TIRConfig()
        self.max_workers = max_workers
        self.patient_map = {}
        self.simulation_configs = []
        self.profile_keys = []

    @staticmethod
    @abstractmethod
    def simulation_loop(virtual_patient_id: str, patient_map: Dict[str, Any]) -> Tuple:
        """
        Run a single patient-controller closed loop simulation.

        This is the core simulation function that must be implemented by subclasses.
        It runs the closed loop for one virtual patient with one parameter profile.

        Args:
            virtual_patient_id: Virtual patient ID (e.g., "adult_001_param_0_NO_MEAL")
            patient_map: Dictionary mapping virtual patient IDs to (patient_name, profile, scenario)

        Returns:
            Tuple of (t, BG, CHO, insulin, target_bg) where:
                - t: Time array
                - BG: Blood glucose measurements
                - CHO: Carbohydrate intake
                - insulin: Insulin delivered
                - target_bg: Target blood glucose value

        Note:
            This must be a static method for multiprocessing pickle compatibility.
        """
        pass

    @abstractmethod
    def generate_patient_map_and_simulation_config(
        self,
        run_dir: Path,
    ) -> Tuple[Dict[str, Tuple[str, Dict[str, Any], Any]], List[Tuple[str, Path]]]:
        """
        Generate mapping from virtual patient ID to (patient_name, profile, scenario).

        This method should implement the logic for creating patient combinations,
        parameter profiles, and scenarios specific to the use case.

        Args:
            run_dir: Directory to save simulation results

        Returns:
            Tuple of (patient_map, simulation_configs)
            - patient_map: Dictionary mapping virtual_patient_id -> (patient_name, profile, scenario)
            - simulation_configs: List of tuples (virtual_patient_id, img_save_dir)
        """
        pass

    @staticmethod
    def _run_single_simulation_wrapper(
        simulation_loop: Callable,
        virtual_patient_id: str,
        img_save_dir: Path,
        tir_config: TIRConfig,
        patient_map: Dict[str, Any],
    ) -> Tuple:
        """
        Wrapper for running a single simulation with error handling and result saving.

        Args:
            simulation_loop: Simulation loop function/static method
            virtual_patient_id: Unique ID for this simulation (contains patient name, param index, and scenario)
            img_save_dir: Directory to save images
            tir_config: TIR configuration
            patient_map: Dictionary mapping virtual patient IDs to (patient_name, profile, scenario)

        Returns:
            Tuple of (virtual_patient_id, img_save_dir, time_in_range_or_error)
        """
        try:
            # Print configuration when simulation starts
            patient_name, profile, scenario = patient_map[virtual_patient_id]
            logger.info(f"\n{'='*80}")
            logger.info(f"🚀 Starting simulation: {virtual_patient_id}")
            logger.info(f"   Patient: {patient_name}")
            logger.info(f"   Scenario: {scenario}")
            logger.info(f"   Profile Configuration:")
            for key, value in profile.items():
                # Format nested dicts nicely
                if isinstance(value, dict):
                    logger.info(f"      {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"         {sub_key}: {sub_value}")
                else:
                    logger.info(f"      {key}: {value}")
            logger.info(f"{'='*80}\n")

            # Call the simulation_loop static method
            result = simulation_loop(
                virtual_patient_id=virtual_patient_id,
                patient_map=patient_map,
            )

            # User returned (t, BG, CHO, insulin, target_bg)
            t, BG, CHO, insulin, target_bg = result

            # Calculate TIR
            time_in_range = tir_config.calculate_time_in_range(BG)

            # Save plot (virtual_patient_id already contains scenario)
            file_name = img_save_dir / f"{virtual_patient_id}.png"

            plot_and_save_with_tir(
                t, BG, CHO, insulin, target_bg, file_name, time_in_range, tir_config
            )

            return (
                virtual_patient_id,
                img_save_dir,
                time_in_range,
            )

        except Exception as e:
            logger.error(f"Simulation failed for {virtual_patient_id}: {str(e)}")
            return (
                virtual_patient_id,
                img_save_dir,
                str(e),  # Error message
            )

    def _save_results(self, results: List[Dict], output_dir: Path):
        """
        Save results to CSV and JSON files.

        Args:
            results: List of result dictionaries
            output_dir: Directory to save results
        """
        # Get TIR category field names based on the current standard
        tir_categories = [cat.value for cat in self.tir_config.get_order()]
        tir_acceptance_categories = [
            f"{cat.value}_acceptable" for cat in self.tir_config.get_order()
        ]

        # Define field order
        fieldnames = [
            "virtual_patient_id",
            *self.profile_keys,
            "error_msg",
            *tir_categories,
            *tir_acceptance_categories,
            "acceptable_count",
            "status",
        ]

        # Enrich results with profile data from patient_map
        enriched_results = []
        for result in results:
            enriched_result = result.copy()
            virtual_patient_id = result["virtual_patient_id"]

            # Get profile from patient_map
            if virtual_patient_id in self.patient_map:
                _, profile, _ = self.patient_map[virtual_patient_id]

                # Add each profile parameter to the result
                for key in self.profile_keys:
                    if key in profile:
                        enriched_result[key] = profile[key]

                # Calculate acceptable count
                patient_type = PatientType.ADULT  # Default
                if "child" in virtual_patient_id.lower():
                    patient_type = PatientType.CHILD
                elif "adolescent" in virtual_patient_id.lower():
                    patient_type = PatientType.ADOLESCENT

                category_results, acceptable_count = (
                    self.tir_config.get_time_in_range_acceptance(
                        time_in_range=result, patient_group=patient_type
                    )
                )

                # Add acceptable count to results
                enriched_result["acceptable_count"] = acceptable_count

                # Add category acceptance results (e.g., "Very Low_acceptable": True/False)
                for category, is_acceptable in category_results.items():
                    enriched_result[f"{category.value}_acceptable"] = is_acceptable

            enriched_results.append(enriched_result)

        # Save to CSV
        try:
            csv_file = output_dir / "results.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in enriched_results:
                    writer.writerow(row)
            logger.info(f"✅ Results saved to {csv_file}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")

        # Save to JSON
        try:
            json_file = output_dir / "results.json"
            with open(json_file, "w") as f:
                json.dump(enriched_results, f, indent=2, default=str)
            logger.info(f"✅ Results saved to {json_file}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")

        # Print summary
        failed_count = sum(
            1 for r in enriched_results if "Failed" in r.get("status", "")
        )
        success_count = sum(
            1
            for r in enriched_results
            if r.get("status") in ["Completed", "Completed_On_Retry"]
        )

        logger.info(f"\n📊 Final Summary:")
        logger.info(f"  ✅ Successful: {success_count}")
        logger.info(f"  ❌ Failed: {failed_count}")
        logger.info(f"  Total: {len(results)}")

    def _execute_parallel_simulations(
        self,
        simulation_configs: List[Tuple],
        is_retry: bool = False,
    ) -> Tuple[List[Dict], List[Tuple]]:
        """
        Execute simulations in parallel.

        Args:
            simulation_configs: List of tuples (virtual_patient_id, img_save_dir)
            is_retry: Whether this is a retry run

        Returns:
            Tuple of (results_list, failed_configs)
        """
        results_list = []
        failed_configs = []

        # Use max_workers if set, otherwise use CPU count
        if self.max_workers is not None:
            num_workers = min(self.max_workers, len(simulation_configs))
        else:
            num_workers = min(multiprocessing.cpu_count(), len(simulation_configs))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all simulations
            futures = []
            for config in simulation_configs:
                (virtual_patient_id, img_save_dir) = config

                future = executor.submit(
                    OpenAPSParameterTuningBase._run_single_simulation_wrapper,
                    self.simulation_loop,  # Pass the simulation_loop static method
                    virtual_patient_id,
                    img_save_dir,
                    self.tir_config,
                    self.patient_map,
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                result = future.result()

                if not is_retry:
                    logger.info(f"Completed: {result[2]}")

                (virtual_patient_id, img_save_dir, time_in_range_or_error) = result

                # Check if it's an error (string) or success (dict)
                if isinstance(time_in_range_or_error, str):
                    status = "Failed_After_Retry" if is_retry else "Failed"
                    error_msg = f"Error: {time_in_range_or_error}"
                    time_in_range_or_error = None

                    if not is_retry:
                        failed_configs.append((virtual_patient_id, img_save_dir))
                else:
                    status = "Completed_On_Retry" if is_retry else "Completed"
                    error_msg = None

                # Build result dict
                result_dict = {
                    "virtual_patient_id": virtual_patient_id,
                    "error_msg": error_msg,
                    "status": status,
                }

                # Add time_in_range results (flatten the dict)
                if time_in_range_or_error:
                    for category, percentage in time_in_range_or_error.items():
                        result_dict[category.value] = percentage

                results_list.append(result_dict)

        return results_list, failed_configs

    def _run_parallel_simulation(
        self,
        output_dir: Path,
        retry_failed: bool = True,
    ) -> List[Dict]:
        """
        Run parallel simulations across patients, profiles, and scenarios.

        Uses the class's simulation_loop static method for running simulations.

        Args:
            output_dir: Directory to save results
            retry_failed: Whether to retry failed simulations

        Returns:
            List of result dictionaries
        """
        # Create output directory with timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generated {len(self.patient_map)} virtual patient IDs")

        # Dump patient_map for reference
        patient_map_file = output_dir / "patient_map.json"
        with open(patient_map_file, "w") as f:
            json.dump(self.patient_map, f, indent=2, default=str)
        logger.info(f"Patient map saved to {patient_map_file}")

        # Determine number of workers
        total_configs = len(self.simulation_configs)
        num_workers = min(multiprocessing.cpu_count(), total_configs)
        logger.info(
            f"🚀 Starting {total_configs} simulations using {num_workers} workers..."
        )

        # Run initial simulations
        results, failed_configs = self._execute_parallel_simulations(
            self.simulation_configs,
            is_retry=False,
        )

        logger.info(f"✅ Initial run completed!")

        # Retry failed simulations
        if retry_failed and failed_configs:
            logger.info(f"🔄 Retrying {len(failed_configs)} failed configurations...")

            retry_results, _ = self._execute_parallel_simulations(
                failed_configs,
                is_retry=True,
            )

            # Update results with retry outcomes
            for retry_result in retry_results:
                for i, original_result in enumerate(results):
                    if (
                        original_result["virtual_patient_id"]
                        == retry_result["virtual_patient_id"]
                        and original_result["status"] == "Failed"
                    ):
                        results[i] = retry_result
                        break

            logger.info(f"✅ Retry completed!")

        # Save results
        self._save_results(results, output_dir)

        return results

    def run(self) -> List[Dict]:
        """
        Execute the parameter tuning experiment.

        This method orchestrates the entire experiment:
        1. Creates output directory
        2. Generates patient map and simulation configs (implementation-specific)
        3. Runs parallel simulations
        4. Saves results

        Returns:
            List of result dictionaries
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / timestamp

        # Generate patient map and simulation configs (subclass implements this)
        self.patient_map, self.simulation_configs = (
            self.generate_patient_map_and_simulation_config(run_dir)
        )

        # Extract profile keys from first patient's profile
        if self.patient_map:
            first_virtual_id = list(self.patient_map.keys())[0]
            self.profile_keys = list(self.patient_map[first_virtual_id][1].keys())
            logger.info(f"Profile keys: {self.profile_keys}")

        # Run parallel simulations using class's simulation_loop static method
        results = self._run_parallel_simulation(
            output_dir=run_dir,
        )

        return results
