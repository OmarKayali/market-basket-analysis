# run_pipeline.py
import pandas as pd
from retail_rec import (
    clean_transactions, build_customer_item_matrix,
    cluster_customers_dbscan, rules_for_cluster, rules_for_all_clusters
)

# 1) Read data & clean
df_raw = pd.read_csv("data.csv", sep=";")
df = clean_transactions(df_raw)  # Keep only 4 columns and clean

# 2) Customer × Item (boolean) -> DBSCAN
cust_item_bin = build_customer_item_matrix(df, binarize=True)
labels, X_pca = cluster_customers_dbscan(
    cust_item_bin,
    n_pca=20, eps=5.0, min_samples=5, metric="euclidean"
)
print(labels.value_counts())

# 3) Run FP-Growth in Cluster 0 (same as notebook example)
rules_c0 = rules_for_cluster(df, labels, cluster_id=0,
                             min_support=0.005, min_confidence=0.3)
rules_c0.to_csv("cluster0_rules_sorted.csv", index=False)

# 4) Optional: Export rules for all clusters in batch
all_rules = rules_for_all_clusters(df, labels, min_support=0.005, min_confidence=0.3)
for cid, r in all_rules.items():
    r.to_csv(f"cluster{cid}_rules_sorted.csv", index=False)