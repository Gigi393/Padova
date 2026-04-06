"""
Patient type definitions for glucose control analytics.

This module defines patient types used across different glucose control
and simulation systems.
"""

from enum import Enum


class PatientType(Enum):
    """
    Enum representing different patient age groups for diabetes management.

    These categories are commonly used in clinical practice and research
    to differentiate treatment parameters and glucose targets.
    """
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    CHILD = "child"
