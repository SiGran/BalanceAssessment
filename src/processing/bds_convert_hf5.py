import h5py
import polars as pl
from pathlib import Path


class BDSConverterH5:
    """Converts the BDS data to a more efficient HDF5 format"""
    def __init__(self, data_dir: Path, h5_path: Path):
        self.data_dir = Path(data_dir)
        self.h5_path = h5_path

    def _parse_info_file(self):
        """Parse the info.txt file containing metadata"""
        info_path = self.data_dir / "BDSinfo.txt"
        metadata = {}

        with open(info_path, "r") as f:
            current_file = None
            for line in f:
                line = line.strip()
                if line.startswith('File:'):
                    current_file = line.split('File:')[1].strip()
                    metadata[current_file] = {}
                elif current_file and ':' in line:
                    key, value = line.split(':', 1)
                    metadata[current_file][key.strip()] = value.strip()
        return metadata

    def convert_to_h5(self):
        """Convert all txt files to a single H5 file"""
        # Get metadata from info file
        metadata = self._parse_info_file()

        # Create H5 file
        with h5py.File(self.h5_path, 'w') as hf:
            # Create groups for data and metadata
            data_grp = hf.create_group('timeseries')
            meta_grp = hf.create_group('metadata')

            # Process each file
            for file_name, meta in metadata.items():
                file_path = self.data_dir / file_name

                if file_path.exists():
                    # Read time series data
                    data = pl.read_csv(
                        file_path,
                        separator=' ',
                        has_header=False
                    ).to_numpy()

                    # Store time series data
                    data_grp.create_dataset(
                        file_name,
                        data=data,
                        compression="gzip",
                        compression_opts=9
                    )

                    # Store metadata
                    meta_ds = meta_grp.create_dataset(file_name, data=[0])
                    for key, value in meta.items():
                        meta_ds.attrs[key] = value

                    print(f"Processed: {file_name}")


# Usage
if __name__ == "__main__":
    converter = BalanceDataConverter(
        data_dir='path/to/your/data/directory',
        h5_path='balance_data.h5'
    )
    converter.convert_to_h5()