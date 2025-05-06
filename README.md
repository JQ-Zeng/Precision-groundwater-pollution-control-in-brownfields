# GBRT-Based Groundwater Pollution Classification

This script applies a Gradient Boosting Classifier to assess groundwater pollution based on integrated inorganic and organic contamination levels.

## Features

- Integrates rule-based and data-driven classification logic
- Uses GBRT for supervised multi-class classification (4 pollution levels)
- Outputs performance metrics, decision tree visualizations, ROC curves, and learning curves
- Saves prediction results in a CSV file

## Requirements

- Python 3.7+
- Libraries: pandas, matplotlib, sklearn, numpy, shap

Install with:

```bash
pip install pandas matplotlib scikit-learn shap numpy
```

## How to Use

1. Place your file `pollution_data.csv` in the directory.
2. Run the script:

```bash
python gbrt_pollution_classifier.py
```

3. Output: `pollution_level_results.csv`, plots for ROC, learning curve, and decision trees.

## Input Data Format

- A CSV file with columns: `inorganic`, `organic`
- Pollution levels will be computed using predefined rules

## Citation

Refer to Supplementary Methods 8 in the paper for detailed method logic.
