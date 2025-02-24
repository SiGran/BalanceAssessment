import h5py
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from pathlib import Path


class AgeGroupPredictor:
    """Predict age groups using src data and best T scores.

    This class implements a binary classifier using Support Vector Machine (SVM)
    to predict whether a subject is Young or Old based on their src data
    (COP distance) and Best T scores.

    Attributes
    ----------
    h5_path : Path
        Path to the HDF5 file containing the src data
    exclude_incomplete : bool
        Whether to exclude subjects with incomplete data
    incomplete_subjects : set[int]
        Set of subject IDs with incomplete data
    model : SVC
        Support Vector Machine classifier with linear kernel
    """
    def __init__(self, h5_path: Path, exclude_incomplete: bool = True):
        """Initialize the age group predictor.

        Parameters
        ----------
        h5_path : Path
            Path to the HDF5 file containing the src data
        exclude_incomplete : bool, optional
            Whether to exclude subjects with incomplete data, by default True
        """
        self.h5_path = h5_path
        self.exclude_incomplete = exclude_incomplete
        self.incomplete_subjects = {59, 60, 86, 122, 134}
        self.model = SVC(kernel='linear')

    def train_and_get_results(self) -> dict[str, tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]:
        """Train models and get prediction results for each condition.

        Trains two SVM models for each condition:
        1. Using only COP distance
        2. Using both COP distance and Best T scores

        Returns
        -------
        dict[str, tuple]
            Dictionary mapping conditions to tuples containing:
            - X_test: Test feature matrix
            - y_test: True test labels
            - y_pred: Predicted labels
            - decision_values: Model decision function values
            - metrics: Dictionary with model performance metrics

        Notes
        -----
        The metrics dictionary contains:
        - accuracy: Classification accuracy
        - f1: F1 score (positive class is 'Young')
        - y_pred: Predicted labels
        - model: Trained SVM model
        """
        with h5py.File(self.h5_path, 'r') as h5_file:
            condition_data = {'All Conditions': []}

            for trial_id in h5_file['timeseries']:
                subject_id = int(h5_file['trials'][trial_id].attrs['subject_id'])

                if self.exclude_incomplete and subject_id in self.incomplete_subjects:
                    continue

                vision = h5_file['trials'][trial_id].attrs['Vision']
                surface = h5_file['trials'][trial_id].attrs['Surface']
                condition = f"{vision} & {surface}"

                trial_data = h5_file['timeseries'][trial_id][:]
                columns = h5_file['timeseries'][trial_id].attrs['columns'].tolist()
                age_group = h5_file['subjects'][str(subject_id)].attrs['AgeGroup']
                best_score = float(h5_file['subjects'][str(subject_id)].attrs['Best_T'])

                df = pl.DataFrame(trial_data, schema=columns)
                df = df.with_columns([
                    pl.lit(age_group).alias('AgeGroup'),
                    pl.lit(subject_id).alias('SubjectID'),
                    pl.lit(best_score).alias('Best_T')
                ])

                if condition not in condition_data:
                    condition_data[condition] = []
                condition_data[condition].append(df)
                condition_data['All Conditions'].append(df)

            results = {}
            for condition, data in condition_data.items():
                df = pl.concat(data)
                df = df.group_by(['SubjectID', 'AgeGroup', 'Best_T']).agg(
                    pl.col('COP_distance[cm]').sum()
                )

                # Create feature matrix with both COP distance and Best_T
                X_cop = df['COP_distance[cm]'].to_numpy().reshape(-1, 1)
                X_best = df['Best_T'].to_numpy().reshape(-1, 1)
                y = df['AgeGroup'].to_numpy()

                # Train and evaluate both models
                metrics = {}
                for features, name in [
                    (X_cop, 'COP only'),
                    (np.hstack([X_cop, X_best]), 'COP + Best_T')
                ]:
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, y, test_size=0.2, random_state=31
                    )

                    model = SVC(kernel='linear')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    decision_values = model.decision_function(X_test)

                    # Store metrics and predictions
                    metrics[name] = {
                        'accuracy': (y_test == y_pred).mean(),
                        'f1': f1_score(y_test, y_pred, pos_label='Young'),
                        'y_pred': y_pred,
                        'model': model
                    }

                    if name == 'COP + Best_T':
                        results[condition] = (
                        X_test, y_test, y_pred, decision_values, metrics)
            return results

    @staticmethod
    def print_model_results(condition_results: dict) -> None:
        """Print performance metrics for each condition and model type.

        Parameters
        ----------
        condition_results : dict
            Dictionary containing prediction results and metrics
            for each condition, as returned by train_and_get_results()

        Returns
        -------
        None
            Prints metrics to standard output
        """
        for condition, (_, _, _, _, metrics) in condition_results.items():
            print(f"\nCondition: {condition}")
            for model_type in ['COP only', 'COP + Best_T']:
                print(f"{model_type}:")
                print(f"  Accuracy: {metrics[model_type]['accuracy']:.2%}")
                print(f"  F1 Score: {metrics[model_type]['f1']:.2f}")
        return