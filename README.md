# SOM-KM Clustering for Groundwater Hydrogeochemistry

This R script implements the Self-Organizing Map (SOM) combined with K-means clustering (SOM-KM) method to analyze and spatially cluster hydrogeochemical data from 379 monitoring wells.

## Features

- Trains a SOM using 14 hydrogeochemical variables
- Applies K-means to the SOM codebook vectors
- Visualizes component planes, mapping results, and cluster boundaries
- Exports cluster assignment for each sample

## Requirements

- R
- Packages: `kohonen`

Install required package if not already installed:

```R
install.packages("kohonen")
```

## How to Use

1. Place your data file named `Groundwater chemical data.csv` in the working directory.
2. Run the script:

```R
source("som_km_clustering.R")
```

3. Output: `sample_clusters_with_ID.csv` will be generated.

## Input Data Format

- CSV file with 1st column as sample ID, and columns 2 to 15 as numeric variables.
- No missing values.

## Citation

Refer to Supplementary Methods 4 in the main article for full parameter setup and explanation.
