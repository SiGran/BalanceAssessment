"""Streamlit application for Balance Data Set (BDS) analysis and visualization.

This module provides an interactive web interface for analyzing and visualizing
src data from the BDS dataset. It includes functionality for:
- Loading and processing src measurement data
- Creating age-binned visualizations
- Displaying trial statistics
- Showing age group prediction results

The application uses:
- Streamlit for the web interface
- Polars for data processing
- Plotly for interactive visualizations
- Custom analysis modules from other parts of the project

Functions
---------
load_and_process_data
    Cache and process BDS data with age binning calculations
run_vis_app
    Set up and run the Streamlit visualization interface

Usage
-----
Run the script with the path to the HDF5 file as an argument:
    python app.py /path/to/BDS.h5

The interface will be available in the default web browser.
"""

import sys
import streamlit as st
import numpy as np
import polars as pl
from pathlib import Path
from plotly.graph_objects import Figure

from ..analyzing import BDSAnalyzer
from ..analyzing import AgeGroupPredictor
from ..visualizing import PlotGenerator

@st.cache_data
def load_and_process_data(h5_path: Path) -> dict[int, pl.DataFrame]:
    """Load and process BDS data with age binning calculations.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file containing BDS data

    Returns
    -------
    dict[int, pl.DataFrame]
        Dictionary mapping number of bins to corresponding DataFrames
        Each DataFrame contains trial data with age binning information

    Notes
    -----
    Age binning is based on quantiles rather than fixed ranges to ensure
    equal number of participants in each bin.
    The function is cached using Streamlit's cache_data decorator.
    """
    with BDSAnalyzer(h5_path) as analyzer:
        df = analyzer.get_path_lengths_by_age()

        # Add condition information
        conditions = []
        for trial_id in df["trial_id"]:
            vision = analyzer.h5_file["trials"][trial_id].attrs["Vision"]
            surface = analyzer.h5_file["trials"][trial_id].attrs["Surface"]
            conditions.append(f"{vision} & {surface}")

        df = df.with_columns([
            pl.Series(name="condition", values=conditions)
        ])

        # Get unique subjects and their ages
        subject_ages = (
            df.group_by("subject_id")
            .agg(pl.col("age").first())
            .sort("age")
        )

        binned_dfs = {}
        for n_bins in range(2, 17):
            # Calculate quantiles for equal number of subjects
            quantiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(subject_ages["age"], quantiles)

            # Create bin labels
            bin_labels = [
                f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
                for i in range(n_bins)
            ]

            # Assign bins
            bin_indices = np.digitize(df["age"], bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            binned_df = df.with_columns([
                pl.Series(name="age_bin", values=[bin_labels[i] for i in bin_indices])
            ])

            binned_dfs[n_bins] = binned_df

        return binned_dfs

def plot_age_prediction_results(h5_path: Path) -> list[Figure]:
    """Create plots for age group prediction results for each condition.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file containing BDS data

    Returns
    -------
    list[Figure]
        List of Plotly figures containing confusion matrices and
        decision boundary plots
    """
    predictor = AgeGroupPredictor(h5_path)
    condition_results = predictor.train_and_get_results()

    plots = []

    # Process 'All Conditions' first
    if 'All Conditions' in condition_results:
        all_cond = condition_results.pop('All Conditions')
        plots.extend(PlotGenerator._create_prediction_plots(
            all_cond, 'All Conditions (Combined Data)'
        ))

    # Process remaining conditions
    for condition, results in condition_results.items():
        plots.extend(PlotGenerator._create_prediction_plots(results, condition))

    return plots


def run_vis_app(h5_path: Path) -> None:
    """Run the Streamlit visualization application.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file containing BDS data

    Notes
    -----
    This function sets up the Streamlit interface with:
    - Data loading and processing
    - Sidebar controls for visualization parameters
    - Interactive plots and statistics display
    - Age group prediction analysis

    The interface includes:
    - Metric selection (Path Length or Average Speed)
    - Trial condition splitting option
    - Age group binning controls
    - Summary statistics toggle
    - Age group prediction results toggle
    """
    st.title("Balance Data Analysis")
    st.markdown("""
        This application analyzes src data from the BDS dataset.
        Use the controls in the sidebar to adjust the visualization.
    """)

    try:
        # Load and process all data at once
        with st.spinner("Loading data..."):
            binned_dfs = load_and_process_data(h5_path)

        # Sidebar controls
        st.sidebar.header("Analysis Controls")

        # Initialize session state
        if "last_n_bins" not in st.session_state:
            st.session_state.last_n_bins = 4

        # Metric selection
        metric = st.sidebar.radio(
            "Select Metric",
            options=["Path Length", "Average Speed"],
            help="Path Length: total distance traveled\nAverage Speed: mean velocity"
        )
        metric_col = "avg_speed" if metric == "Average Speed" else "path_length"
        metric_label = "Speed (cm/s)" if metric == "Average Speed" else "Path Length (cm)"

        # Split by condition checkbox
        split_conditions = st.sidebar.checkbox(
            "Split by trial conditions",
            help="Show separate box plots for each trial condition"
        )

        # Age binning controls
        n_bins = st.sidebar.slider(
            "Number of age groups",
            min_value=2,
            max_value=16,
            value=st.session_state.last_n_bins
        )
        st.session_state.last_n_bins = n_bins

        # Move checkboxes to sidebar
        st.sidebar.markdown("---")  # Add a visual separator
        show_stats = st.sidebar.checkbox("Show summary statistics")
        show_prediction = st.sidebar.checkbox("Show Age Group Prediction Results")

        # Create and show plot with progress indicator
        with st.spinner("Creating visualization..."):
            plot_df = binned_dfs[n_bins]  # Get the correct binned DataFrame
            fig = PlotGenerator.plot_path_lengths(
                plot_df,
                metric_col,
                split_conditions,
                n_bins,
                metric_label
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        if show_stats:
            st.write("### Summary Statistics by Age Group")
            if split_conditions:
                summary = plot_df.group_by(["age_bin", "condition"]).agg([
                    pl.col(metric_col).mean().alias(f"mean_{metric_col}"),
                    pl.col(metric_col).std().alias(f"std_{metric_col}"),
                    pl.col(metric_col).count().alias("n_trials"),
                    pl.col("age").mean().alias("mean_age"),
                    pl.n_unique("subject_id").alias("n_subjects")
                ])
            else:
                summary = plot_df.group_by("age_bin").agg([
                    pl.col(metric_col).mean().alias(f"mean_{metric_col}"),
                    pl.col(metric_col).std().alias(f"std_{metric_col}"),
                    pl.col(metric_col).count().alias("n_trials"),
                    pl.col("age").mean().alias("mean_age"),
                    pl.n_unique("subject_id").alias("n_subjects")
                ])
            st.write(summary)

        if show_prediction:
            st.write("### Age Group Prediction Analysis")
            with st.spinner("Calculating age group prediction results..."):
                plots = PlotGenerator.plot_age_prediction_results(h5_path)

                for i in range(0, len(plots), 4):  # Changed from 2 to 4
                    # First row: confusion matrices
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plots[i], use_container_width=True)
                    with col2:
                        st.plotly_chart(plots[i + 1], use_container_width=True)

                    # Second row: decision plots
                    col3, col4 = st.columns(2)
                    with col3:
                        st.plotly_chart(plots[i + 2], use_container_width=True)
                    with col4:
                        st.plotly_chart(plots[i + 3], use_container_width=True)

    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.exception(e)


if __name__ == "__main__":
    try:
        h5_path = Path(sys.argv[1])
        run_vis_app(h5_path)
    except IndexError:
        st.error("No H5 file path provided!")
        st.write("Please check the h5_path variable in main.py and it's location")
        st.write("The H5 file should be created when running main.py")
    except Exception as e:
        st.error(f"Error: {str(e)}")