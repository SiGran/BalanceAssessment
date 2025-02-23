"""Balance Data Set (BDS) Analysis Application.

This application handles the processing, analysis, and visualization of BDS data.
It converts raw data to HDF5 format, validates the data, performs age group
predictions, and launches an interactive visualization dashboard.

Author: Simon Grannetia
Created: 2025-02-22
Version: 1.0.0

Usage:
    python main.py

Requirements:
    - BDS data directory at ../data/BDS/
    - streamlit
    - polars
    - h5py
    - sklearn
    - plotly
"""
import subprocess
import sys
import json

from pathlib import Path

from processing.bds_convert_hf5 import BDSConverterH5
from analyzing.bds_analyzer import BDSAnalyzer
from analyzing.age_group_predictor import AgeGroupPredictor


def main():
    # Set up paths
    bds_dir = Path.cwd() / "data" / "BDS"
    h5_path = Path.cwd() / "data" / "BDS.h5"
    vis_app = Path.cwd() / "src" / "visualizing" / "app.py"

    # Check if data directory exists, try alternate paths
    if not bds_dir.exists():
        alternate_bds_dir = Path.cwd().parent / "data" / "BDS"
        if alternate_bds_dir.exists():
            bds_dir = alternate_bds_dir
            h5_path = Path.cwd().parent / "data" / "BDS.h5"
        else:
            print(f"Error: Data directory not found at {bds_dir}")
            print(f"Also tried: {alternate_bds_dir}")
            print("Please check the README for setup instructions.")
            sys.exit(1)

    if not vis_app.exists():
        alternate_vis_app = Path.cwd() / "visualizing" / "app.py"
        if alternate_vis_app.exists():
            vis_app = alternate_vis_app
        else:
            print(f"Error: Visualization app not found at {vis_app}")
            print(f"Also tried: {alternate_vis_app}")
            print(f"Expected path: {vis_app.absolute()}")
            sys.exit(1)

    try:
        # Create H5 file if needed
        if not h5_path.exists():
            print("Converting data to H5 format...")
            with BDSConverterH5(bds_dir, h5_path) as converter:
                converter.convert_to_h5()
            print("Conversion complete!")

        # Validate the data
        print("\nValidating data...")
        with BDSAnalyzer(h5_path) as analyzer:
            validation = analyzer.validate_data()
            print("Data validation results:")
            print(json.dumps(validation, indent=4))

            # Train and evaluate age group predictor
        # Train and evaluate age group predictor
        print("\nEvaluating age group predictor...")
        for exclude in [True, False]:
            print(f"\n{'Without' if exclude else 'With'} incomplete subjects:")
            predictor = AgeGroupPredictor(h5_path, exclude_incomplete=exclude)
            results = predictor.train_and_get_results()
            predictor.print_model_results(results)


        # Launch visualization
        print("\nLaunching visualization app...")
        result = subprocess.run(
            ["streamlit", "run", str(vis_app), "--", str(h5_path)],
            check=True,
            cwd=str(Path.cwd())  # Set working directory to src
        )

        if result.returncode != 0:
            print("Error: Visualization app failed to launch")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()