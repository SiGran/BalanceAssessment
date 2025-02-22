import sys
import streamlit as st
import numpy as np
import polars as pl
from pathlib import Path

from src.analyzing.bds_analyzer import BDSAnalyzer


@st.cache_data
def load_and_process_data(h5_path: Path):
    """Cache both data loading and binning calculations"""
    with BDSAnalyzer(h5_path) as analyzer:
        df = analyzer.get_path_lengths_by_age()

        # Precalculate all bin options
        min_age = df["age"].min()
        max_age = df["age"].max()
        age_array = df["age"].to_numpy()

        binned_dfs = {}
        for n_bins in range(2, 9):  # 2 through 8 bins
            bin_edges = np.linspace(min_age, max_age, n_bins + 1)
            bin_labels = [
                f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
                for i in range(n_bins)
            ]

            bin_indices = np.digitize(age_array, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            binned_df = df.with_columns([
                pl.Series(name="age_bin", values=[bin_labels[i] for i in bin_indices])
            ])

            binned_dfs[n_bins] = binned_df

        return binned_dfs


def run_vis_app(h5_path: Path):
    """Run the Streamlit visualization app

    Parameters
    ----------
    h5_path : Path
        Path to the H5 file containing the balance data
    """
    st.title("Balance Data Analysis")
    st.markdown("""
        This application analyzes balance data from the BDS dataset.
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
            max_value=8,
            value=st.session_state.last_n_bins
        )
        st.session_state.last_n_bins = n_bins

        # Get pre-calculated DataFrame for this number of bins
        plot_df = binned_dfs[n_bins]

        # Create and show plot with progress indicator
        with st.spinner("Creating visualization..."):
            with BDSAnalyzer(h5_path) as analyzer:
                fig = analyzer.plot_path_lengths(
                    plot_df,
                    metric_col,
                    split_conditions,
                    n_bins,
                    metric_label
                )
            st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        if st.checkbox("Show summary statistics"):
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

            st.write("### Summary Statistics by Age Group")
            st.write(summary)

            # Add download button
            if st.checkbox("Download Data"):
                csv = plot_df.to_csv()
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="balance_data.csv",
                    mime="text/csv"
                )

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