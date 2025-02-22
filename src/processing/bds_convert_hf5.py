import h5py
import polars as pl
from pathlib import Path


class BDSConverterH5:
    """Converts the BDS data to a more efficient HDF5 format

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
        """Set up HDF5 file and groups when entering context"""
        self.h5_file = h5py.File(self.h5_path, "w")
        self.subjects_grp = self.h5_file.create_group("subjects")
        self.trials_grp = self.h5_file.create_group("trials")
        self.data_grp = self.h5_file.create_group("timeseries")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close HDF5 file when exiting context"""
        if hasattr(self, "h5_file"):
            self.h5_file.close()
        return

    def _store_attribute(self, dataset: h5py.Dataset, key: str, value) -> None:
        """Helper method to store attributes with proper types

        Parameters
        ----------
        dataset : h5py.Dataset
            The HDF5 dataset to store the attribute in
        key : str
            The name of the attribute
        value : Any
            The value to store
        """
        if value is None:  # Handle null values
            dataset.attrs[key] = "NA"
        elif isinstance(value, (int, float)):  # Keep numeric types as is
            dataset.attrs[key] = value
        elif isinstance(value, bool):  # Keep booleans as is
            dataset.attrs[key] = value
        else:  # Convert everything else to string
            dataset.attrs[key] = str(value)
        return

    def _process_subject(
            self,
            subject_id: str,
            trial_row: dict,
            trials_grp: h5py.Group,
            data_grp: h5py.Group
    ) -> None:
        """Process and store trial data and corresponding time series data for a subject.

        Parameters
        ----------
        subject_id : str
            The identifier of the subject
        trial_row : dict
            Dictionary containing trial metadata from a single row
        trials_grp : h5py.Group
            HDF5 group where trial metadata will be stored
        data_grp : h5py.Group
            HDF5 group where time series data will be stored

        Notes
        -----
        Time series data contains measurements of forces (F), moments (M),
        and center of pressure (COP) over time.

        Raises
        ------
        FileNotFoundError
            If the time series file for the trial is not found
        """
        trial_id = trial_row["Trial"]

        try:
            # Store trial conditions
            trial_ds = trials_grp.create_dataset(trial_id, data=[0])
            trial_ds.attrs["subject_id"] = str(subject_id)

            # Store trial conditions
            for col in self.trial_columns:
                self._store_attribute(trial_ds, col, trial_row[col])

            # Store time series data
            file_path = self.data_dir / f"{trial_id}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Time series file not found: {file_path}")

            # Read time series data with headers
            df = pl.read_csv(
                file_path,
                separator="\t",
                has_header=True
            )

            # Create dataset with column information
            ds = data_grp.create_dataset(
                trial_id,
                data=df.to_numpy(),
                compression=self.compression,
                compression_opts=self.compression_opts
            )

            # Store column names from DataFrame
            ds.attrs["columns"] = df.columns

            print(f"Processed: {trial_id}")

        except Exception as e:
            print(f"Error processing trial {trial_id}: {str(e)}")
        return


    def convert_to_h5(self) -> None:
        """Convert BDS data to HDF5 format with efficient metadata storage.

        This function:
        1. Reads the metadata from BDSinfo.txt
        2. Creates an HDF5 file with three main groups:
            - subjects: containing subject-specific data that remains constant
            - trials: containing trial conditions (Vision, Surface)
            - timeseries: containing the actual balance measurement data
        3. For each subject:
            - Stores constant subject data (demographics, health info, etc.)
            - Processes all trials for that subject
            - Links trials to subjects and their time series data

        Notes
        -----
        The HDF5 structure is organized as:
        /subjects/
            subject_id/
                attributes: Age, Gender, Height, etc.
        /trials/
            trial_id/
                attributes: subject_id, Vision, Surface
        /timeseries/
            trial_id/
                data: numpy array of measurements

        The file is automatically closed after processing due to the context manager.

        Raises
        ------
        FileNotFoundError
            If BDSinfo.txt or any time series file is not found
        """
        # Read metadata
        metadata_df = pl.read_csv(
            self.data_dir / "BDSinfo.txt",
            separator="\t"
        )

        # All other columns (except Trial) are constant per subject
        constant_columns = [
            col for col in metadata_df.columns
            if col not in self.trial_columns + ["Trial"]
        ]

        # Get unique subjects
        unique_subjects = metadata_df["Subject"].unique()

        for subject_id in unique_subjects:
            # Get first record for this subject
            subject_data = metadata_df.filter(
                pl.col("Subject") == subject_id
            ).to_dicts()[0]

            # Store subject's constant data
            subject_ds = self.subjects_grp.create_dataset(
                str(subject_id),
                data=[0]
            )
            for col in constant_columns:
                self._store_attribute(subject_ds, col, subject_data[col])

            # Get all trials for this subject
            subject_trials = metadata_df.filter(
                pl.col("Subject") == subject_id
            )

            # Store each trial's data
            for trial_row in subject_trials.iter_rows(named=True):
                self._process_subject(
                    subject_id,
                    trial_row,
                    self.trials_grp,
                    self.data_grp
                )

        return
