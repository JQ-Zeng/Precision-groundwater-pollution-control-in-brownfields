# DNN-Based Groundwater Pollution Classification

This script uses Deep Neural Networks (DNNs) to classify groundwater pollution levels based on 20 heavy metal concentrations, integrating the Chinese Groundwater Quality Standard (GB/T 14848-2017).

## Features

- Classifies pollutants into pollution grades using national standards
- Trains a DNN model to predict pollution level from heavy metal data
- Visualizes loss curves, feature importance, prediction performance, and 3D spatial distribution
- Exports final classification result for all sites

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, matplotlib, seaborn, sklearn, keras

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras
```

## How to Use

1. Place your data file `Groundwater pollution data DNN.csv` in the same directory.
2. Run the script:

```bash
python dnn_groundwater_classifier.py
```

3. Output: `final_pollution_classification.csv` and related plots will be generated.

## Citation

Refer to Supplementary Methods 5 for model workflow and pollution grading thresholds.
