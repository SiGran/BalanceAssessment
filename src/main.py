from pathlib import Path
import polars as pl

from processing.bds_convert_hf5 import BDSConverterH5
from analyzing.bds_analyzer import BDSAnalyzer

if __name__ == '__main__':
    bds_dir = Path.cwd().parent / "data" / "BDS" # Path to directory
    h5_path = Path.cwd().parent / "data" / "BDS.h5" # Path to more efficient H5 file with all the data
    # TODO: add note to readme for if path don't work
    # read in files and convert to H5 if it doesn't exist
    if not h5_path.exists():

        with BDSConverterH5(bds_dir, h5_path) as converter:
            converter.convert_to_h5()

    with BDSAnalyzer(h5_path) as analyzer:
        # Validate data
        validation = analyzer.validate_data()
        print("Validation results:", validation)

        # Create plot
        fig = analyzer.plot_path_lengths_by_age()
        fig.show()

        # Get summary statistics
        analyzer.get_path_lengths_by_age()
