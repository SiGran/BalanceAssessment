"""Utility module for generating plots in the BDS analysis application.

This module provides plotting functionality for visualizing:
- Age-binned balance measurements
- Trial condition comparisons
- Age group prediction results
- Confusion matrices and decision boundaries

The module uses Plotly Express for interactive visualizations.
"""

import plotly.express as px
from plotly.graph_objects import Figure
import polars as pl
import numpy as np
from pathlib import Path

from src.analyzing.age_group_predictor import AgeGroupPredictor


class PlotGenerator:
    """Generate interactive plots for BDS data analysis.

    This class provides static methods for creating various types of plots
    used in the BDS analysis application, including box plots for age groups
    and visualization of machine learning results.
    """

    @staticmethod
    def plot_path_lengths(
            df: pl.DataFrame,
            metric_col: str,
            split_conditions: bool,
            n_bins: int,
            metric_label: str
    ) -> Figure:
        """Create box plot of path lengths or speeds grouped by age.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing trial data with age binning
        metric_col : str
            Column name for the metric to plot (path_length or avg_speed)
        split_conditions : bool
            Whether to split plots by trial conditions
        n_bins : int
            Number of age groups used for binning
        metric_label : str
            Label for the y-axis of the plot

        Returns
        -------
        Figure
            Plotly figure object containing the box plot
        """
        unique_bins = sorted(df["age_bin"].unique())

        if split_conditions:
            fig = px.box(
                df,
                x="age_bin",
                y=metric_col,
                color="condition",
                title=f"{metric_label} by Age and Condition ({n_bins} groups)",
                labels={
                    "age_bin": "Age Range",
                    metric_col: metric_label,
                    "condition": "Trial Condition"
                },
                category_orders={"age_bin": unique_bins}
            )
        else:
            fig = px.box(
                df,
                x="age_bin",
                y=metric_col,
                title=f"{metric_label} by Age ({n_bins} groups)",
                labels={
                    "age_bin": "Age Range",
                    metric_col: metric_label
                },
                category_orders={"age_bin": unique_bins}
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

    @staticmethod
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

    @staticmethod
    def _create_prediction_plots(
            results: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict],
            condition: str
    ) -> list[Figure]:
        """Create confusion matrices and decision plots for both models.

        Parameters
        ----------
        results : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
            Tuple containing:
            - X_test: Test feature matrix
            - y_test: True test labels
            - y_pred: Predicted labels
            - decision_values: Model decision function values
            - metrics: Dictionary of model metrics
        condition : str
            Trial condition description

        Returns
        -------
        list[Figure]
            List of Plotly figures for confusion matrices and decision plots
        """
        X_test, y_test, y_pred, decision_values, metrics = results
        plots = []

        # Create plots for both models
        for model_type in ['COP only', 'COP + Best_T']:
            y_pred_model = y_pred if model_type == 'COP + Best_T' else metrics[model_type]['y_pred']

            # Create confusion matrix
            confusion_counts = np.zeros((2, 2))
            for true_idx, pred_idx in zip(
                    (y_test == 'Young').astype(int),
                    (y_pred_model == 'Young').astype(int)
            ):
                confusion_counts[true_idx, pred_idx] += 1

            metrics_text = (
                f"<br>Accuracy: {metrics[model_type]['accuracy']:.2%}"
                f"<br>F1 Score: {metrics[model_type]['f1']:.2f}"
            )

            fig_confusion = px.imshow(
                confusion_counts,
                labels=dict(x="Predicted", y="True", color="Count"),
                x=['Young', 'Old'],
                y=['Young', 'Old'],
                color_continuous_scale="Viridis",
                title=f"{model_type} - {condition}{metrics_text}"
            )
            plots.append(fig_confusion)

            if model_type == 'COP only':
                features = X_test[:, [0]]  # Only use first feature for COP only
                x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
                y_min, y_max = x_min, x_max
                scatter_y = features[:, 0]
            else:
                features = X_test  # Use both features for COP + Best_T
                x_min, x_max = features[:, 0].min() - 1, features[:,
                                                         0].max() + 1  # COP Distance range
                y_min, y_max = features[:, 1].min() - 1, features[:,
                                                         1].max() + 1  # Best_T range
                scatter_y = features[:, 1]

            fig_decision = px.scatter(
                x=features[:, 0],
                y=scatter_y,
                color=y_test,
                labels={
                    'x': 'COP Distance',
                    'y': 'Best_T Score' if model_type == 'COP + Best_T' else 'COP Distance',
                    'color': 'Age Group'
                },
                title=f"Decision Plot - {model_type} - {condition}"
            )

            # Add decision boundary if model is available
            if 'model' in metrics[model_type]:
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100)
                )

                if model_type == 'COP only':
                    Z = metrics[model_type]['model'].decision_function(
                        xx.ravel().reshape(-1, 1)
                    )
                else:
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    Z = metrics[model_type]['model'].decision_function(grid_points)

                Z = Z.reshape(xx.shape)

                fig_decision.add_contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=Z,
                    name='Decision Boundary',
                    showscale=False,
                    colorscale='RdBu',
                    opacity=0.4
                )
            plots.append(fig_decision)

        return plots