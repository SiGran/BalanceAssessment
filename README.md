

# Balance Data Set Analysis

A Python application for analyzing and visualizing human balance data using 
Support Vector Machine (SVM) classification. The application processes data from the 
[Balance Data Set](https://figshare.com/articles/dataset/A_public_data_set_of_quantitative_and_qualitative_evaluations_of_human_balance/3394432/2).

## Features

- Data conversion from raw text files to HDF5 format
- data validation and data balance check (prints to terminal as json)
- Age group prediction using SVM models
- Interactive visualization dashboard including:
  - Age-binned computed COP distance box and whisker plots
    - Age binning done by quantiles, not by age brackets to account for non-uniform age distribution
  - Trial condition comparisons (i.e. difference between vision and surface between trials)
  - Prediction results visualization
  - Statistical summaries

## Installation

### Requirements
- Python 3.11.2 (for sure, maybe lower versions)
- Poetry (recommended) or pip

### Using Poetry (Recommended)

1. Install Poetry following the [official guide](https://python-poetry.org/docs/#installation)
note: More info on using poetry can be found [here](https://realpython.com/dependency-management-python-poetry/#export-dependencies-from-poetry)
3. Clone this repository
3. Navigate to the project directory
4. Install dependencies:
```bash
poetry install
```
5. Copy the BDS data files to the `data/BDS/` directory

### Using pip

If Poetry installation fails, use pip with the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data/
│   └── BDS/              # Raw balance data files
├── src/
│   ├── analyzing/        # Analysis modules
s│   ├── processing/       # Data processing
│   └── visualizing/      # Visualization components
├── poetry.lock          # Poetry dependency lock
├── pyproject.toml       # Project configuration
└── requirements.txt     # Pip requirements
```

## Usage

There are two ways to run the application:

### Using PyCharm (Recommended)
1. Open the project in PyCharm
2. PyCharm will detect the `pyproject.toml` and set up the Poetry environment automatically
3. Run main.py from pycharm.

### Using Command Line
1. From the project root directory:
```bash
poetry install
poetry run python src/main.py
````

2. The application will:
   - Convert raw data to HDF5 format (if needed)
   - Validate data integrity
   - Train and evaluate age group predictors
   - Launch the visualization dashboard

3. Access the dashboard in your browser at `http://localhost:8501`

## Model Description

The application uses two SVM models for age group prediction:
1. COP only: Uses Center of Pressure distance
2. COP + Best T: Combines COP distance with Best T scores

Performance metrics include:
- Accuracy
- F1 Score
- Confusion matrices
- Decision boundary visualizations

## Model evaluation

Using the Euclidean distances computed from the ML (y) and AP (x) axis, the
model has reasonable accuracy in most cases (from 50% to 80%, most often >70%).

Judging from the Box and Whisker plots this seems like a reasonable result: from the plots you clearly see the path 
length is increasing with age. At every age, however, you have people who perform quite well on
the firm surface. This makes it hard for the SVM to achieve very high accuracy and F1 scores.
The Best T scores likely have less of this overlap for the best performers and therefore improve the accuracy and F1 scores.

## Acknowledgments

- Balance Data Set: Santos and Duarte (2016)
- Built with [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), and [scikit-learn](https://scikit-learn.org/)
```
