# README

## Project Overview
The goal of this project is to analyze retail transaction data by performing **customer clustering (K-Means)** and **Market Basket Analysis (FP-Growth association rules)**. This helps uncover purchasing patterns and generate product recommendations for different customer segments.

The pipeline consists of two major steps:
1. Data preprocessing & clustering (K-Means with MaxAbsScaler)
2. Market basket analysis for each cluster (FP-Growth)

---

## Environment Setup
### Dependencies
- Python 3.9+ (preferably via Anaconda)
- pandas  
- numpy  
- scikit-learn  
- mlxtend  
- matplotlib  

Install via conda (recommended):
```bash
conda install -c conda-forge mlxtend scikit-learn -y
```

---

## File Structure
- `data.csv` - raw retail dataset  
- `MBA-KMeans.ipynb` - exploratory notebook with K-Means implementation
- `4990 1.ipynb` - additional exploratory notebook
- `retail_rec.py` - core functions (data cleaning, customer-item matrix, K-Means/DBSCAN clustering, FP-Growth rules)  
- `run_pipeline.py` - main script to execute the full pipeline (CSV output)
- `data_processor_final.py` - full pipeline for web application (JSON output)
- `app.py` - Flask backend API server
- `frontend/index.html` - interactive web dashboard
- `cluster0_rules_sorted.csv` - output file containing rules for Cluster 0 (sorted by lift/confidence)  

---

## Usage

### Option 1: Basic Pipeline (CSV Output)
1. Place `data.csv` in the same directory as the scripts  
2. Run the pipeline:
   ```bash
   python run_pipeline.py
   ```
3. Outputs:
   - Terminal will show cluster sizes (Cluster 0, 1, 2)  
   - CSV files for each cluster: `cluster0_rules_sorted.csv`, `cluster1_rules_sorted.csv`, etc.

### Option 2: Web Application (Recommended)
1. Process the data:
   ```bash
   python data_processor_final.py
   ```
   This generates JSON files in the `output/` folder.

2. Start the Flask backend:
   ```bash
   python app.py
   ```
   Server runs on `http://localhost:5000`

3. Open `frontend/index.html` in your browser for the interactive dashboard.  

---

## Configuration

### Adjusting K-Means Parameters
You can modify the number of clusters in `run_pipeline.py` or `data_processor_final.py`:

```python
labels, X_scaled = cluster_customers_kmeans(
    cust_item_bin,
    n_clusters=3,  # Change this value
    scaler="maxabs",
    random_state=42,
    n_init=20
)
```

### Tuning Association Rules
Experiment with `min_support` and `min_confidence` parameters to explore stronger item associations:

```python
rules_c0 = rules_for_cluster(df, labels, cluster_id=0,
                             min_support=0.005,  # Adjust threshold
                             min_confidence=0.3)  # Adjust threshold
```

---

## Next Steps
- Analyze cluster characteristics to understand customer segments
- Integrate rules into a recommendation system  
- Experiment with different `n_clusters`, `min_support`, and `min_confidence` values  
