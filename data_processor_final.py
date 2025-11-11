import pandas as pd
import numpy as np
import json
import os
from retail_rec import (
    clean_transactions, 
    build_customer_item_matrix,
    cluster_customers_dbscan, 
    rules_for_cluster
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

print("="*50)
print("PROCESSING YOUR RETAIL DATA")
print("="*50)

# =====================================================
# STEP 1: Load and Clean Data
# =====================================================
print("\n1. Loading data from data.csv...")
df_raw = pd.read_csv("data.csv", sep=";", dtype={'BillNo': str})
print(f"   Raw data: {len(df_raw):,} rows")

df = clean_transactions(df_raw)
print(f"   Cleaned data: {len(df):,} rows, {df['CustomerID'].nunique():,} customers")

# =====================================================
# STEP 2: DBSCAN Clustering
# =====================================================
print("\n2. Performing DBSCAN clustering...")
cust_item_bin = build_customer_item_matrix(df, binarize=True)

# Get the standardized and PCA transformed data for visualization
X = cust_item_bin.values.astype(float)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Cluster
labels, _ = cluster_customers_dbscan(
    cust_item_bin,
    n_pca=20, 
    eps=5.0, 
    min_samples=5, 
    metric="euclidean"
)

cluster_counts = labels.value_counts()
print("   Cluster distribution:")
for cluster_id in sorted(cluster_counts.index):
    print(f"     Cluster {cluster_id}: {cluster_counts[cluster_id]:,} customers")

# Save cluster data with PCA coordinates
cluster_data = []
for idx, cluster_id in labels.items():
    loc = cust_item_bin.index.get_loc(idx)
    cluster_data.append({
        'customer_id': int(idx),
        'cluster': int(cluster_id),
        'pca_x': float(X_pca[loc, 0]),
        'pca_y': float(X_pca[loc, 1])
    })

with open('output/clusters.json', 'w') as f:
    json.dump(cluster_data, f)

# =====================================================
# STEP 3: FP-Growth on Cluster 0
# =====================================================
print("\n3. Running FP-Growth on Cluster 0...")
rules_c0 = rules_for_cluster(
    df, labels, 
    cluster_id=0,
    min_support=0.005, 
    min_confidence=0.3
)
print(f"   Found {len(rules_c0):,} association rules")

# Convert frozensets to lists for JSON
rules_data = []
for _, row in rules_c0.iterrows():
    rules_data.append({
        'antecedent': list(row['antecedents']),
        'consequent': list(row['consequents']),
        'support': float(row['support']),
        'confidence': float(row['confidence']),
        'lift': float(row['lift'])
    })

with open('output/rules.json', 'w') as f:
    json.dump(rules_data, f)

# =====================================================
# STEP 4: Generate Frequent Itemsets
# =====================================================
print("\n4. Extracting frequent itemsets...")
from mlxtend.frequent_patterns import fpgrowth

# Get cluster 0 customers
cust_ids = labels.index[labels == 0]
df_c0 = df[df['CustomerID'].isin(cust_ids)]

# Build basket
basket = (df_c0.groupby(['BillNo', 'Itemname'])['Quantity']
          .sum().unstack().fillna(0))
basket_bin = basket > 0

# FP-Growth for itemsets
itemsets = fpgrowth(basket_bin, min_support=0.005, use_colnames=True)
itemsets['count'] = (itemsets['support'] * len(basket)).astype(int)

itemsets_data = []
for _, row in itemsets.iterrows():
    itemsets_data.append({
        'items': list(row['itemsets']),
        'support': float(row['support']),
        'count': int(row['count'])
    })

with open('output/itemsets.json', 'w') as f:
    json.dump(itemsets_data, f)

# =====================================================
# STEP 5: Statistics
# =====================================================
print("\n5. Generating statistics...")

# Top items
top_items = df['Itemname'].value_counts().head(10)
top_items_data = [
    {'name': item, 'count': int(count)} 
    for item, count in top_items.items()
]

# Cluster stats
cluster_stats = {}
for cluster_id in sorted(labels.unique()):
    cluster_stats[str(cluster_id)] = {
        'count': int(cluster_counts[cluster_id]),
        'label': f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
    }

stats = {
    'total_transactions': int(df['BillNo'].nunique()),
    'total_customers': int(df['CustomerID'].nunique()),
    'total_itemsets': len(itemsets),
    'total_rules': len(rules_c0),
    'total_clusters': len(cluster_counts),
    'top_items': top_items_data,
    'cluster_stats': cluster_stats
}

with open('output/statistics.json', 'w') as f:
    json.dump(stats, f)

# =====================================================
# STEP 6: Products List
# =====================================================
print("\n6. Creating product list...")
products = sorted(df['Itemname'].unique().tolist())

with open('output/products.json', 'w') as f:
    json.dump({'products': products}, f)

# =====================================================
# STEP 7: Top Products per Cluster
# =====================================================
print("\n7. Analyzing top products per cluster...")

# Add cluster labels to customer_item matrix
cust_item_with_cluster = cust_item_bin.copy()
cust_item_with_cluster['Cluster'] = labels

cluster_products = {}
for cluster_id in sorted(labels.unique()):
    cluster_data_temp = cust_item_with_cluster[
        cust_item_with_cluster['Cluster'] == cluster_id
    ].drop('Cluster', axis=1)
    
    top_items_cluster = cluster_data_temp.sum().sort_values(ascending=False).head(10)
    
    cluster_products[str(cluster_id)] = [
        {'item': item, 'count': int(count)} 
        for item, count in top_items_cluster.items()
    ]

with open('output/cluster_products.json', 'w') as f:
    json.dump(cluster_products, f)

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "="*50)
print("âœ… DATA PROCESSING COMPLETE!")
print("="*50)
print(f"Total Transactions: {stats['total_transactions']:,}")
print(f"Total Customers: {stats['total_customers']:,}")
print(f"Frequent Itemsets: {stats['total_itemsets']:,}")
print(f"Association Rules: {stats['total_rules']:,}")
print(f"Clusters Found: {stats['total_clusters']}")
print("\nFiles created in 'output/' folder:")
print("  ✓ clusters.json")
print("  ✓ itemsets.json")
print("  ✓ rules.json")
print("  ✓ statistics.json")
print("  ✓ products.json")
print("  ✓ cluster_products.json")
print("\nReady to start Flask backend!")
print("="*50)