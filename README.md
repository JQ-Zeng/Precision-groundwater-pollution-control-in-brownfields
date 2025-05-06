# DMVC: Deep Multi-View Clustering for Groundwater Pollution Analysis

This script applies a Deep Multi-View Clustering (DMVC) model to integrate and analyze five types of multi-source data: land use, hydrochemistry, vulnerability, mobility, and pollution grade.

## Features

- Uses a deep encoder to extract latent representations from multi-view data
- Applies KMeans clustering in the latent space
- Outputs cluster assignments and feature summaries
- Visualizes results with PCA

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, matplotlib, scikit-learn, tensorflow

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## How to Use

1. Place your file `Groundwater class.csv` in the directory.
2. Run the script:

```bash
python dmvc_multiview_clustering.py
```

3. Outputs:
- `pollution_repair_clusters.csv` – cluster assignment for each sample
- `cluster_summary.csv` – average view features per cluster
- PCA plot of cluster separation

## Input Format

CSV with columns: `vulnerability`, `mobility`, `hydrochemistry`, `land use`, `pollution`.

## Citation

See Supplementary Methods 9 in the original article for full explanation of DMVC model logic.
