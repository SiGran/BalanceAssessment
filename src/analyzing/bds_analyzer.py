"""BDS Data Analysis Module.

This module provides functionality for analyzing Balance Data Set (BDS) data
stored in HDF5 format. It includes methods for data validation, path length
calculations, and age-based grouping of src measurements.

The module assumes the following HDF5 structure:
/subjects/
    subject_id/
        attributes: Age, AgeGroup
/trials/
    trial_id/
        attributes: subject_id, Vision, Surface
/timeseries/
    trial_id/
        data: measurements including COP_distance[cm]
"""

import h5py
import polars as pl
from pathlib import Path

from ..CONSTANTS import DT


class BDSAnalyzer:
    """Analyze Balance Data Set (BDS) data from HDF5 file.

    This class provides methods for validating the data structure,
    calculating Center of Pressure (COP) path lengths, and analyzing
    src measurements across different age groups.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file containing BDS data

    Attributes
    ----------
    h5_path : Path
        Path to the HDF5 file
    h5_file : h5py.File
        Open HDF5 file (only available within context manager)
    """
    def __init__(self, h5_path: Path):
        """Initialize analyzer with path to HDF5 file."""
        self.h5_path = Path(h5_path)

    def __enter__(self) -> "BDSAnalyzer":
        """Open HDF5 file when entering context.

        Returns
        -------
        BDSAnalyzer
            The analyzer instance with opened HDF5 file
        """
        self.h5_file = h5py.File(self.h5_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close HDF5 file when exiting context.

        Parameters
        ----------
        exc_type : type
            Type of the raised exception, if any
        exc_val : Exception
            Instance of the raised exception, if any
        exc_tb : traceback
            Traceback of the raised exception, if any
        """
        if hasattr(self, "h5_file"):
            self.h5_file.close()

    def validate_data(self) -> dict:
        """Validate the HDF5 file structure and contents.

        Checks for:
        - Number of participants
        - Number of trials per participant
        - Total number of trials
        - Distribution of trial conditions
        - Age group src
        - Missing data by subject

        Returns
        -------
        dict
            Dictionary containing validation results:
            - missing_trials_total: Number of missing trials
            - subjects_with_missing_data: Details of missing trials by subject
            - age_group_balance: Distribution of subjects across age groups
        """
        # Expected values
        expected = {
            "participants": 163,
            "trials_per_participant": 12,
            "total_trials": 163 * 12,
            "conditions": {
                "Closed & Firm": 3,
                "Open & Firm": 3,
                "Closed & Foam": 3,
                "Open & Foam": 3
            }
        }

        # Actual counts
        actual = {
            "participants": len(self.h5_file["subjects"]),
            "total_trials": len(self.h5_file["trials"])
        }

        # Age group src check
        age_groups = {"Young": 0, "Old": 0}
        incomplete_age_groups = {"Young": 0, "Old": 0}
        incomplete_subjects = {59, 60, 86, 122, 134}

        for subject_id in self.h5_file["subjects"]:
            age_group = self.h5_file["subjects"][subject_id].attrs["AgeGroup"]
            if int(subject_id) in incomplete_subjects:
                incomplete_age_groups[age_group] += 1
            else:
                age_groups[age_group] += 1

        # Find incomplete subjects (rest of the existing code)
        subjects_conditions = {}
        for trial_id in self.h5_file["trials"]:
            subject_id = self.h5_file["trials"][trial_id].attrs["subject_id"]
            vision = self.h5_file["trials"][trial_id].attrs["Vision"]
            surface = self.h5_file["trials"][trial_id].attrs["Surface"]
            condition = f"{vision} & {surface}"

            if subject_id not in subjects_conditions:
                subjects_conditions[subject_id] = {}

            subjects_conditions[subject_id][condition] = subjects_conditions[
                                                             subject_id].get(condition,
                                                                             0) + 1

        # Check which subjects have missing trials
        missing_data = {}
        for subject_id, conditions in subjects_conditions.items():
            missing_conditions = {}
            for condition, expected_count in expected["conditions"].items():
                actual_count = conditions.get(condition, 0)
                if actual_count < expected_count:
                    missing_conditions[condition] = expected_count - actual_count

            if missing_conditions:
                missing_data[subject_id] = missing_conditions

        return {
            "missing_trials_total": expected["total_trials"] - actual["total_trials"],
            "subjects_with_missing_data": missing_data,
            "age_group_balance": {
                "with_incomplete": {
                    "Young": age_groups["Young"] + incomplete_age_groups["Young"],
                    "Old": age_groups["Old"] + incomplete_age_groups["Old"]
                },
                "without_incomplete": age_groups
            }
        }

    def _get_cop_path_length(self, trial_id: str) -> tuple[float, float]:
        """Calculate COP path length and average speed for a trial.

        Uses pre-calculated point-to-point distances stored in the HDF5 file
        to compute total path length and average speed.

        Parameters
        ----------
        trial_id : str
            The trial identifier in the HDF5 file

        Returns
        -------
        tuple[float, float]
            Total path length (cm), average speed (cm/s)

        Notes
        -----
        Average speed is calculated by dividing mean point-to-point
        distance by the time interval DT
        """
        # Get data
        data = self.h5_file["timeseries"][trial_id][:]
        columns = self.h5_file["timeseries"][trial_id].attrs["columns"].tolist()

        # Create DataFrame and get pre-calculated distances
        df = pl.DataFrame(data, schema=columns)
        distances = df["COP_distance[cm]"]

        return distances.sum(), distances.mean() / DT

    def get_path_lengths_by_age(self) -> pl.DataFrame:
        """Calculate path lengths for all trials grouped by age.

        Creates a DataFrame containing path lengths and speeds for all trials,
        along with subject age and trial identifiers.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - trial_id: Trial identifier
            - subject_id: Subject identifier
            - age: Subject's age
            - path_length: Total COP path length (cm)
            - avg_speed: Average COP speed (cm/s)
        """
        results = []

        for trial_id in self.h5_file["trials"]:
            # Get subject info
            subject_id = self.h5_file["trials"][trial_id].attrs["subject_id"]
            age = float(self.h5_file["subjects"][subject_id].attrs["Age"])

            # Calculate path length
            path_length, avg_speed = self._get_cop_path_length(trial_id)

            results.append({
                "trial_id": trial_id,
                "subject_id": subject_id,
                "age": age,
                "path_length": path_length,
                "avg_speed": avg_speed
            })

        return pl.DataFrame(results)
