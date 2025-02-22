# bds_analyzer.py
import h5py
import polars as pl
import numpy as np
import plotly.express as px
from pathlib import Path

from src.CONSTANTS import DT


class BDSAnalyzer:
    """Analyze BDS data from HDF5 file"""
    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)

    def __enter__(self):
        """Open HDF5 file when entering context"""
        self.h5_file = h5py.File(self.h5_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file when exiting context"""
        if hasattr(self, "h5_file"):
            self.h5_file.close()

    def validate_data(self) -> dict:
        """Validate the HDF5 file structure and contents"""
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

        # Find incomplete subjects
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
            "subjects_with_missing_data": missing_data
        }

    def _calculate_cop_path_length(self, trial_id: str) -> tuple[float, float]:
        """Calculate COP path length and average speed for a trial using stored distances

        Parameters
        ----------
        trial_id : str
            The trial identifier

        Returns
        -------
        tuple[float, float]
            Total path length (cm), average speed (cm/s)
        """
        # Get data
        data = self.h5_file["timeseries"][trial_id][:]
        columns = self.h5_file["timeseries"][trial_id].attrs["columns"].tolist()

        # Create DataFrame and get pre-calculated distances
        df = pl.DataFrame(data, schema=columns)
        distances = df["COP_distance[cm]"]

        return distances.sum(), distances.mean() / DT

    def get_path_lengths_by_age(self) -> pl.DataFrame:
        """Calculate path lengths for all trials and group by age"""
        results = []

        for trial_id in self.h5_file["trials"]:
            # Get subject info
            subject_id = self.h5_file["trials"][trial_id].attrs["subject_id"]
            age = float(self.h5_file["subjects"][subject_id].attrs["Age"])

            # Calculate path length
            path_length, avg_speed = self._calculate_cop_path_length(trial_id)

            results.append({
                "trial_id": trial_id,
                "subject_id": subject_id,
                "age": age,
                "path_length": path_length,
                "avg_speed": avg_speed  # Add this
            })

        df = pl.DataFrame(results)

        # Calculate summary statistics including age info
        summary = df.group_by("age").agg([
            pl.col("path_length").mean().alias("mean_path_length"),
            pl.col("path_length").std().alias("std_path_length"),
            pl.col("path_length").count().alias("n_trials"),
            pl.n_unique("subject_id").alias("n_subjects")
        ])

        print("\nSummary statistics:")
        print(summary)

        return df

    def plot_path_lengths(self, df: pl.DataFrame, metric_col: str,
                          split_conditions: bool,
                          n_bins: int, metric_label: str) -> px.box:
        """Create interactive box plot of data

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing the data to plot
        metric_col : str
            Column name to plot (path_length or avg_speed)
        split_conditions : bool
            Whether to split by trial conditions
        n_bins : int
            Number of age groups shown
        metric_label : str
            Label for the metric being plotted

        Returns
        -------
        px.box
            Plotly box plot figure
        """
        # Sort the age bins
        unique_bins = sorted(df["age_bin"].unique())

        if split_conditions:
            fig = px.box(
                df.to_pandas(),
                x="age_bin",
                y=metric_col,
                color="condition",
                title=f"{metric_label} by Age and Condition ({n_bins} groups)",
                labels={
                    "age_bin": "Age Range",
                    metric_col: metric_label,
                    "condition": "Trial Condition"
                },
                category_orders={"age_bin": unique_bins}  # Ensure consistent order
            )
        else:
            fig = px.box(
                df.to_pandas(),
                x="age_bin",
                y=metric_col,
                title=f"{metric_label} by Age ({n_bins} groups)",
                labels={
                    "age_bin": "Age Range",
                    metric_col: metric_label
                },
                category_orders={"age_bin": unique_bins}  # Ensure consistent order
            )

        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            showlegend=True
        )

        return fig
