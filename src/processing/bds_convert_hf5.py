"""Convert Balance Data Set (BDS) to HDF5 format.

This module provides functionality to convert the raw BDS text files into a more
efficient HDF5 format. It handles both metadata and time series data, organizing
them into a structured format with proper grouping and compression.

The module implements a class that processes:
- Subject metadata from BDSinfo.txt
- Trial conditions (Vision and Surface)
- Time series data from individual trial files

The resulting HDF5 file structure is:
/subjects/
    subject_id/
        attributes: Age, Gender, Height, etc.
/trials/
    trial_id/
        attributes: subject_id, Vision, Surface
/timeseries/
    trial_id/
        data: numpy array of measurements
"""

import h5py
import polars as pl
from pathlib import Path


class BDSConverterH5:
    """Convert BDS data to HDF5 format with metadata and time series.

    This class handles the conversion of raw BDS text files into a structured
    HDF5 format. It implements a context manager for proper file handling
    and provides methods for processing subjects, trials, and time series data.

    Parameters
    ----------
    data_dir : Path
        Directory containing the BDS data files
    h5_path : Path
        Path where the HDF5 file will be saved
    compression : str, optional
        Compression method for HDF5, by default "gzip"
    compression_opts : int, optional
        Compression level, by default 9

    Attributes
    ----------
    data_dir : Path
        Path to the directory containing BDS data
    h5_path : Path
        Path where the HDF5 file will be saved
    compression : str
        HDF5 compression method
    compression_opts : int
        Compression level for HDF5 datasets
    trial_columns : list
        Columns that vary by trial (Vision, Surface)
    """

    def __init__(
            self,
            data_dir: Path,
            h5_path: Path,
            compression: str = "gzip",
            compression_opts: int = 9
    ):
        self.data_dir = Path(data_dir)
        self.h5_path = h5_path
        self.compression = compression
        self.compression_opts = compression_opts
        self.trial_columns = ["Vision", "Surface"]
        return

    def __enter__(self) -> "BDSConverterH5":
        """Create HDF5 file and groups when entering context.

        Returns
        -------
        BDSConverterH5
            The converter instance with initialized HDF5 groups

        Notes
        -----
        Creates three main groups in the HDF5 file:
        - subjects: For subject metadata
        - trials: For trial conditions
        - timeseries: For measurement data
        """
        self.h5_file = h5py.File(self.h5_path, "w")
        self.subjects_grp = self.h5_file.create_group("subjects")
        self.trials_grp = self.h5_file.create_group("trials")
        self.data_grp = self.h5_file.create_group("timeseries")
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
        return

    def _store_attribute(self, dataset: h5py.Dataset, key: str, value) -> None:
        """Store attribute with proper type handling.

        Parameters
        ----------
        dataset : h5py.Dataset
            HDF5 dataset to store the attribute in
        key : str
            Attribute name
        value : Any
            Value to store

        Notes
        -----
        Type handling:
        - None -> "NA"
        - Numbers -> kept as is
        - Booleans -> kept as is
        - Others -> converted to string
        """
        if value is None:
            dataset.attrs[key] = "NA"
        elif isinstance(value, (int, float)):
            dataset.attrs[key] = value
        elif isinstance(value, bool):
            dataset.attrs[key] = value
        else:
            dataset.attrs[key] = str(value)
        return

    def _calculate_cop_distance(self, df: pl.DataFrame) -> pl.Series:
        """Calculate center of pressure (COP) distance between consecutive points.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with COPx[cm] and COPy[cm] columns

        Returns
        -------
        pl.Series
            Series containing point-to-point COP distances

        Notes
        -----
        Uses Euclidean distance formula: sqrt(dx² + dy²)
        First point distance is set to 0
        """
        dx = df["COPx[cm]"].diff().fill_null(0)
        dy = df["COPy[cm]"].diff().fill_null(0)
        distances = (dx ** 2 + dy ** 2).sqrt()
        return distances

    def _process_subject(
            self,
            subject_id: str,
            trial_row: dict,
            trials_grp: h5py.Group,
            data_grp: h5py.Group
    ) -> None:
        """Process trial data and time series for a subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        trial_row : dict
            Trial metadata from BDSinfo.txt
        trials_grp : h5py.Group
            HDF5 group for trial metadata
        data_grp : h5py.Group
            HDF5 group for time series data

        Notes
        -----
        For each trial:
        1. Stores trial conditions
        2. Links trial to subject
        3. Processes time series data
        4. Calculates COP distances
        5. Stores data with compression

        Raises
        ------
        FileNotFoundError
            If time series file is not found
        """
        trial_id = trial_row["Trial"]

        try:
            # Store trial metadata
            trial_ds = trials_grp.create_dataset(trial_id, data=[0])
            trial_ds.attrs["subject_id"] = str(subject_id)

            for col in self.trial_columns:
                self._store_attribute(trial_ds, col, trial_row[col])

            # Process time series data
            file_path = self.data_dir / f"{trial_id}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Time series file not found: {file_path}")

            df = pl.read_csv(file_path, separator="\t", has_header=True)
            cop_distance = self._calculate_cop_distance(df)
            df = df.with_columns(cop_distance.alias("COP_distance[cm]"))

            # Store time series with compression
            ds = data_grp.create_dataset(
                trial_id,
                data=df.to_numpy(),
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            ds.attrs["columns"] = df.columns

            print(f"Processed: {trial_id}")

        except Exception as e:
            print(f"Error processing trial {trial_id}: {str(e)}")
        return

    def convert_to_h5(self) -> None:
        """Convert BDS data to HDF5 format.

        Reads metadata from BDSinfo.txt and processes all subjects and their
        trials, storing data in a structured HDF5 format with compression.

        Notes
        -----
        Processing steps:
        1. Read metadata from BDSinfo.txt
        2. Process each subject:
           - Store constant subject data
           - Process all trials
           - Store time series data
        3. Close file (handled by context manager)

        Raises
        ------
        FileNotFoundError
            If BDSinfo.txt or any time series file is not found
        """
        metadata_df = pl.read_csv(self.data_dir / "BDSinfo.txt", separator="\t")
        constant_columns = [
            col for col in metadata_df.columns
            if col not in self.trial_columns + ["Trial"]
        ]

        for subject_id in metadata_df["Subject"].unique():
            # Store subject metadata
            subject_data = metadata_df.filter(
                pl.col("Subject") == subject_id
            ).to_dicts()[0]

            subject_ds = self.subjects_grp.create_dataset(str(subject_id), data=[0])
            for col in constant_columns:
                self._store_attribute(subject_ds, col, subject_data[col])

            # Process subject's trials
            subject_trials = metadata_df.filter(pl.col("Subject") == subject_id)
            for trial_row in subject_trials.iter_rows(named=True):
                self._process_subject(
                    subject_id,
                    trial_row,
                    self.trials_grp,
                    self.data_grp
                )
        return