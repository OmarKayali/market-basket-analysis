# K-Means Migration Summary

## Overview
The project has been successfully migrated from **DBSCAN** clustering to **K-Means** clustering while maintaining all Market Basket Analysis functionality.

## What Changed

### 1. Core Functions (`retail_rec.py`)
- **Added**: `cluster_customers_kmeans()` function
  - Uses `MaxAbsScaler` for feature scaling (default)
  - Supports `StandardScaler` as alternative
  - Default: 3 clusters, 20 initializations
  - Returns cluster labels (0, 1, 2) - no noise cluster like DBSCAN (-1)
- **Kept**: `cluster_customers_dbscan()` function for backwards compatibility

### 2. Pipeline Scripts

#### `run_pipeline.py`
- Replaced `cluster_customers_dbscan()` with `cluster_customers_kmeans()`
- Parameters:
  - `n_clusters=3`
  - `scaler="maxabs"`
  - `random_state=42`
  - `n_init=20`
- Changed `exclude_noise=False` in `rules_for_all_clusters()` (K-Means has no noise)

#### `data_processor_final.py`
- Replaced DBSCAN clustering with K-Means
- Uses PCA (2 components) for visualization only (not for clustering)
- Removed "Noise" label handling in cluster statistics

### 3. Documentation

#### `README.md`
- Updated project description to mention K-Means
- Clarified file structure and usage options
- Added configuration section for adjusting K-Means parameters
- Updated output descriptions (no more cluster -1)

## Key Differences: DBSCAN vs K-Means

| Feature | DBSCAN | K-Means |
|---------|--------|---------|
| **Cluster Count** | Automatic (density-based) | Fixed (`n_clusters=3`) |
| **Noise Points** | Yes (cluster -1) | No (all assigned) |
| **Preprocessing** | StandardScaler + PCA (20 components) | MaxAbsScaler only |
| **Parameters** | eps, min_samples | n_clusters, n_init |
| **Cluster IDs** | -1, 0, 1, 2, ... | 0, 1, 2 |

## Benefits of K-Means

1. **Simpler**: Fewer parameters to tune (just `n_clusters`)
2. **Guaranteed assignment**: Every customer belongs to a cluster
3. **Faster**: More efficient for large datasets
4. **Interpretable**: Fixed number of segments for business analysis

## Usage

### Quick Start
```bash
# Process data
python data_processor_final.py

# Start backend
python app.py

# Open frontend/index.html in browser
```

### Adjusting Number of Clusters

Edit `data_processor_final.py` or `run_pipeline.py`:

```python
labels, X_scaled = cluster_customers_kmeans(
    cust_item_bin,
    n_clusters=5,  # Change from 3 to 5 clusters
    scaler="maxabs",
    random_state=42,
    n_init=20
)
```

### Alternative: Using Standard Scaler

```python
labels, X_scaled = cluster_customers_kmeans(
    cust_item_bin,
    n_clusters=3,
    scaler="standard",  # Use StandardScaler instead of MaxAbsScaler
    random_state=42,
    n_init=20
)
```

## Compatibility Notes

- **DBSCAN still available**: The original `cluster_customers_dbscan()` function remains in `retail_rec.py`
- **FP-Growth unchanged**: Market Basket Analysis logic is identical
- **API compatible**: Flask backend and frontend work without changes
- **Output format**: JSON and CSV files maintain the same structure

## Testing

After migration, verify:
1. Cluster counts are positive integers (0, 1, 2)
2. All customers are assigned to clusters (no -1 labels)
3. Association rules are generated for each cluster
4. Web dashboard displays clusters correctly

## Next Steps

1. **Experiment with cluster count**: Try different values of `n_clusters` (e.g., 2, 4, 5)
2. **Analyze cluster profiles**: Examine top products per cluster to understand customer segments
3. **Optimize parameters**: Tune `min_support` and `min_confidence` for better rules
4. **Compare results**: Run DBSCAN again to compare clustering quality

## Rollback Instructions

To revert to DBSCAN:

1. In `run_pipeline.py` and `data_processor_final.py`, replace:
```python
from retail_rec import cluster_customers_kmeans
labels, X_scaled = cluster_customers_kmeans(...)
```

with:

```python
from retail_rec import cluster_customers_dbscan
labels, X_pca = cluster_customers_dbscan(
    cust_item_bin,
    n_pca=20, 
    eps=5.0, 
    min_samples=5, 
    metric="euclidean"
)
```

2. Update cluster visualization to use `X_pca` instead of applying PCA separately

