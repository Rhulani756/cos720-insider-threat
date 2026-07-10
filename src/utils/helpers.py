"""
src/utils/helpers.py
--------------------
Shared utility functions and sample profiles.
Sample profiles are derived from actual records in the dataset
to ensure realistic and accurate demonstration behaviour.
"""

import os
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_sample_profiles() -> list:
    """
    Sample employee profiles for prototype demonstration.
    Derived from actual dataset records to ensure predictions
    are realistic and consistent with model behaviour.
    """
    return [
        {
            "name": "Normal Employee — Low Risk",
            "employee_department": "Engineering Department",
            "employee_campus": "Campus C",
            "employee_position": "Design Engineer",
            "employee_seniority_years": 22,
            "is_contractor": 0,
            "employee_classification": 2,
            "has_foreign_citizenship": 0,
            "has_criminal_record": 0,
            "has_medical_history": 0,
            "employee_origin_country": "Georgia",
            "total_printed_pages": 0,
            "num_printed_pages_off_hours": 0,
            "total_files_burned": 4,
            "burned_from_other": 0,
            "is_abroad": 0,
            "trip_day_number": 0.0,
            "hostility_country_level": 0,
            "num_entries": 1,
            "num_unique_campus": 1,
            "late_exit_flag": 0,
            "entry_during_weekend": 1,
        },
        {
            "name": "Malicious Insider — Critical Risk",
            "employee_department": "Engineering Department",
            "employee_campus": "Campus A",
            "employee_position": "Test Engineer",
            "employee_seniority_years": 19,
            "is_contractor": 1,
            "employee_classification": 2,
            "has_foreign_citizenship": 1,
            "has_criminal_record": 0,
            "has_medical_history": 0,
            "employee_origin_country": "Israel",
            "total_printed_pages": 116,
            "num_printed_pages_off_hours": 44,
            "total_files_burned": 93,
            "burned_from_other": 1,
            "is_abroad": 0,
            "trip_day_number": 0.0,
            "hostility_country_level": 0,
            "num_entries": 3,
            "num_unique_campus": 1,
            "late_exit_flag": 0,
            "entry_during_weekend": 1,
        },
        {
            "name": "Borderline Employee — High Risk",
            "employee_department": "Information Technology",
            "employee_campus": "Campus A",
            "employee_position": "Data Scientist",
            "employee_seniority_years": 6,
            "is_contractor": 0,
            "employee_classification": 2,
            "has_foreign_citizenship": 0,
            "has_criminal_record": 0,
            "has_medical_history": 0,
            "employee_origin_country": "UK",
            "total_printed_pages": 60,
            "num_printed_pages_off_hours": 20,
            "total_files_burned": 30,
            "burned_from_other": 0,
            "is_abroad": 0,
            "trip_day_number": 0.0,
            "hostility_country_level": 0,
            "num_entries": 3,
            "num_unique_campus": 1,
            "late_exit_flag": 0,
            "entry_during_weekend": 0,
        },
    ]


def format_confidence_color(confidence: float, prediction: str) -> str:
    if prediction == "Malicious":
        return "#A32D2D" if confidence >= 70 else "#854F0B"
    return "#3B6D11"