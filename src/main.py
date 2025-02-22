# main.py
import subprocess
import sys
import json

from pathlib import Path

from processing.bds_convert_hf5 import BDSConverterH5
from analyzing.bds_analyzer import BDSAnalyzer
from visualizing.app import run_vis_app

def main():
    # Set up paths
    bds_dir = Path.cwd().parent / "data" / "BDS"
    h5_path = Path.cwd().parent / "data" / "BDS.h5"
    vis_app = Path.cwd() / "visualizing" / "app.py"

    # Check if data directory exists
    if not bds_dir.exists():
        print(f"Error: Data directory not found at {bds_dir}")
        print("Please check the README for setup instructions.")
        sys.exit(1)

    if not vis_app.exists():
        print(f"Error: Visualization app not found at {vis_app}")
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

        # Launch visualization
        print("\nLaunching visualization app...")
        run_vis_app(h5_path)
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