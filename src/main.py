from pathlib import Path

from processing.bds_convert_hf5 import BDSConverterH5


if __name__ == '__main__':
    bds_dir = Path.cwd().parent / "data" / "BDS" # Path to directory
    h5_path = Path.cwd().parent / "data" / "BDS.h5" # Path to more efficient H5 file with all the data
    # TODO: add note to readme for if path don't work
    # read in files and convert to H5 if it doesn't exist
    if not h5_path.exists():
        converter = BDSConverterH5(bds_dir, h5_path)
        converter.convert_to_h5()
