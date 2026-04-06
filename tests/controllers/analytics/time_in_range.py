"""
Time in Range (TIR) configuration and standards.

This module defines different TIR standards and thresholds for glucose control analytics.
It can be used independently with any glucose monitoring system.
"""

from enum import Enum
from .patient_types import PatientType


class TIRCategory(Enum):
    """Time in Range category names."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    TARGET = "target"
    LOW = "low"
    VERY_LOW = "very_low"


class TIRStandard(Enum):
    """Available Time in Range standards."""

    BASIC = "basic"  # Basic 4-category standard
    CLINICAL = "clinical"  # Clinical 5-category standard (with very_low)


class TIRConfig:
    """Configuration for Time in Range standards."""

    # Define threshold boundaries for each standard (class-level constants)
    THRESHOLDS = {
        TIRStandard.BASIC: {
            TIRCategory.VERY_HIGH: 250,  # > 250
            TIRCategory.HIGH: 180,  # 180 - 250
            TIRCategory.TARGET: 70,  # 70 - 180
            TIRCategory.LOW: 0,  # 0 - 70
        },
        TIRStandard.CLINICAL: {
            TIRCategory.VERY_HIGH: 250,  # > 250
            TIRCategory.HIGH: 180,  # 180 - 250
            TIRCategory.TARGET: 70,  # 70 -180
            TIRCategory.LOW: 54,  # 54 -70
            TIRCategory.VERY_LOW: 0,  # 0 - 54
        },
    }

    # Define colors for visualization (class-level constants)
    COLORS = {
        TIRCategory.VERY_HIGH: "#FF6B35",
        TIRCategory.HIGH: "#FFB347",
        TIRCategory.TARGET: "#32CE13",
        TIRCategory.LOW: "#DB2020",
        TIRCategory.VERY_LOW: "#8B0000",
    }

    # Define category order (bottom to top for stacked bar) (class-level constants)
    ORDER = {
        TIRStandard.BASIC: [
            TIRCategory.LOW,
            TIRCategory.TARGET,
            TIRCategory.HIGH,
            TIRCategory.VERY_HIGH,
        ],
        TIRStandard.CLINICAL: [
            TIRCategory.VERY_LOW,
            TIRCategory.LOW,
            TIRCategory.TARGET,
            TIRCategory.HIGH,
            TIRCategory.VERY_HIGH,
        ],
    }

    # Refer to paper https://www.nejm.org/doi/full/10.1056/NEJMoa2203913
    # Acceptable ranges from paper (mean±SD) as percentages (0-100)
    # children aged 7-15, adult aged 16-70
    # | Patient Group | Extreme High (>250) level 2 hyperglycemia | High (180-250) level 1 hyperglycemia | Target (70-180) (SD) | Low (<70) hypoglycemia |
    # | ------------- | ----------------------------------------- | ------------------------------------ | -------------------- | ---------------------- |
    # | Children      | 9.3±6.0                                   | 21.1±6.8                             | 67.5±11.5            | 2.1±1.5                |
    # | Adult         | 5.6±4.9                                   | 18.2±8.4                             | 74.5±11.9            | 1.6±2.1                |

    ACCEPTABLE_RANGES = {
        PatientType.CHILD: {
            TIRCategory.VERY_HIGH: (9.3 - 6.0, 9.3 + 6.0),  # mean±SD
            TIRCategory.HIGH: (21.1 - 6.8, 21.1 + 6.8),  # mean±SD
            TIRCategory.TARGET: (67.5 - 11.5, 67.5 + 11.5),  # mean±SD
            TIRCategory.LOW: (2.1 - 1.5, 2.1 + 1.5),  # mean±SD
        },
        PatientType.ADULT: {
            TIRCategory.VERY_HIGH: (5.6 - 4.9, 5.6 + 4.9),  # mean±SD
            TIRCategory.HIGH: (18.2 - 8.4, 18.2 + 8.4),  # mean±SD
            TIRCategory.TARGET: (74.5 - 11.9, 74.5 + 11.9),  # mean±SD
            TIRCategory.LOW: (1.6 - 2.1, 1.6 + 2.1),  # mean±SD
        },
    }

    def __init__(self, standard: TIRStandard = TIRStandard.BASIC):
        """
        Initialize TIRConfig with a specific standard.

        Args:
            standard: The TIR standard to use (defaults to BASIC)
        """
        self.standard = standard

    def get_thresholds(self):
        """Get thresholds for this instance's standard."""
        return self.THRESHOLDS[self.standard]

    @staticmethod
    def get_color(category: TIRCategory):
        """Get color for a given category (static, same for all standards)."""
        return TIRConfig.COLORS[category]

    def get_order(self):
        """Get category order for this instance's standard."""
        return self.ORDER[self.standard]

    @staticmethod
    def get_acceptable_ranges(patient_group: PatientType):
        """Get acceptable ranges for a given patient group (static, only for BASIC standard)."""
        return TIRConfig.ACCEPTABLE_RANGES.get(
            patient_group, TIRConfig.ACCEPTABLE_RANGES[PatientType.ADULT]
        )

    def calculate_time_in_range(self, BG_values) -> dict:
        """
        Calculate time in range statistics for blood glucose values.
        Uses this instance's standard.

        Args:
            BG_values: List of blood glucose readings in mg/dL

        Returns:
            Dictionary with percentages (0-100) for each range category
        """
        thresholds = self.get_thresholds()

        if self.standard == TIRStandard.BASIC:
            time_in_range = {
                TIRCategory.VERY_HIGH: 0,
                TIRCategory.HIGH: 0,
                TIRCategory.TARGET: 0,
                TIRCategory.LOW: 0,
            }

            for bg in BG_values:
                if bg > thresholds[TIRCategory.VERY_HIGH]:
                    time_in_range[TIRCategory.VERY_HIGH] += 1
                elif bg > thresholds[TIRCategory.HIGH]:
                    time_in_range[TIRCategory.HIGH] += 1
                elif bg > thresholds[TIRCategory.TARGET]:
                    time_in_range[TIRCategory.TARGET] += 1
                else:
                    time_in_range[TIRCategory.LOW] += 1

        else:  # TIRStandard.CLINICAL
            time_in_range = {
                TIRCategory.VERY_HIGH: 0,
                TIRCategory.HIGH: 0,
                TIRCategory.TARGET: 0,
                TIRCategory.LOW: 0,
                TIRCategory.VERY_LOW: 0,
            }

            for bg in BG_values:
                if bg > thresholds[TIRCategory.VERY_HIGH]:
                    time_in_range[TIRCategory.VERY_HIGH] += 1
                elif bg > thresholds[TIRCategory.HIGH]:
                    time_in_range[TIRCategory.HIGH] += 1
                elif bg > thresholds[TIRCategory.TARGET]:
                    time_in_range[TIRCategory.TARGET] += 1
                elif bg > thresholds[TIRCategory.LOW]:
                    time_in_range[TIRCategory.LOW] += 1
                else:
                    time_in_range[TIRCategory.VERY_LOW] += 1

        # Convert to percentages (0-100), only include non-zero values
        total = len(BG_values)
        time_in_range = {
            k: (v / total) * 100 for k, v in time_in_range.items() if v > 0
        }

        return time_in_range

    def get_time_in_range_acceptance(
        self,
        time_in_range,
        patient_group: PatientType,
    ) -> tuple:
        """
        Check if time in range values are within acceptable clinical ranges.
        This instance must use BASIC standard.

        Args:
            time_in_range: Dict with time in range statistics (as percentages 0-100)
            patient_group: PatientType enum

        Returns:
            Tuple of (category_results, acceptable_count) where:
            - category_results: Dict[TIRCategory, bool] indicating if each category is acceptable
            - acceptable_count: int count of categories within acceptable ranges

        Raises:
            ValueError: If this instance's standard is not TIRStandard.BASIC
        """
        if self.standard != TIRStandard.BASIC:
            raise ValueError(
                f"is_time_in_range_acceptable only supports TIRStandard.BASIC. "
                f"Current standard: {self.standard}"
            )

        # Get acceptable ranges for the patient group
        acceptable_ranges = self.get_acceptable_ranges(patient_group)

        # Check each category and track results
        category_results = {}
        acceptable_count = 0

        for category, (min_val, max_val) in acceptable_ranges.items():
            if category not in time_in_range:
                category_results[category] = None
                continue

            is_acceptable = min_val <= time_in_range[category] <= max_val
            category_results[category] = is_acceptable

            if is_acceptable:
                acceptable_count += 1

        return (category_results, acceptable_count)
