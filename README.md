# 📘 README

## Project Overview
The goal of this project is to analyze retail transaction data by performing **customer clustering (DBSCAN)** and **Market Basket Analysis (FP-Growth association rules)**. This helps uncover purchasing patterns and generate product recommendations for different customer segments.

The pipeline consists of two major steps:
1. Data preprocessing & clustering (DBSCAN + PCA)
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
- `data.csv` → raw retail dataset  
- `4990 1.ipynb` → exploratory notebook used for testing the pipeline  
- `retail_rec.py` → core functions (data cleaning, customer-item matrix, DBSCAN clustering, FP-Growth rules)  
- `run_pipeline.py` → main script to execute the full pipeline  
- `cluster0_rules_sorted.csv` → output file containing rules for Cluster 0 (sorted by lift/confidence)  

---

## Usage
1. Place `data.csv` in the same directory as the scripts  
2. Run the pipeline:
   ```bash
   python run_pipeline.py
   ```
3. Outputs:
   - Terminal will show cluster sizes (Cluster 0, 1, 2, -1)  
   - `cluster0_rules_sorted.csv` will be generated (extendable to all clusters)  

---

## Extended Features
Enable batch export of rules for all clusters by editing `run_pipeline.py`:

```python
all_rules = rules_for_all_clusters(df, labels, min_support=0.005, min_confidence=0.3)
for cid, r in all_rules.items():
    r.to_csv(f"cluster{cid}_rules_sorted.csv", index=False)
```

---

## Next Steps
- Integrate `clusterX_rules_sorted.csv` into frontend or recommendation system.  
- Experiment with `min_support` / `min_confidence` parameters to explore stronger item associations.  
