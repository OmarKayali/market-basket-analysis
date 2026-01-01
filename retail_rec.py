# retail_rec.py
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Optional

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from mlxtend.frequent_patterns import fpgrowth, association_rules


# ---------- Data Cleaning ----------
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected input columns: BillNo, CustomerID, Itemname, Quantity (+ optional extra columns)
    Cleaning steps: remove duplicates, remove negatives (returns), drop missing (CustomerID/Itemname)
    Return: cleaned table with only ['BillNo','CustomerID','Itemname','Quantity']
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df[df["Quantity"] >= 0]
    df = df.dropna(subset=["CustomerID", "Itemname"])
    return df[["BillNo", "CustomerID", "Itemname", "Quantity"]]


# ---------- Customer-Item Matrix ----------
def build_customer_item_matrix(df: pd.DataFrame, binarize: bool = True) -> pd.DataFrame:
    """
    Build Customer x Item matrix; if binarize=True then convert to True/False (whether purchased or not)
    Row index: CustomerID; Column index: Itemname
    """
    mat = (df.groupby(["CustomerID", "Itemname"])["Quantity"]
             .sum().unstack().fillna(0))
    return (mat > 0) if binarize else mat


# ---------- k-distance curve helper ----------
def k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Return k-distance vector sorted in ascending order (used to manually find eps elbow point)
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    kdist = np.sort(dists[:, -1])
    return kdist


# ---------- K-Means Clustering ----------
def cluster_customers_kmeans(
    customer_item_binary: pd.DataFrame,
    n_clusters: int = 3,
    scaler: str = "maxabs",
    random_state: int = 42,
    n_init: int = 20,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Input: Customer x Item (boolean) matrix
    Steps: scale -> KMeans clustering
    Return: (labels, X_scaled)
        labels: pd.Series(index=CustomerID, name='Cluster', dtype=int)
        X_scaled: scaled features (can be used for PCA visualization)
    """
    X = customer_item_binary.values.astype(float)

    # Scale the data
    if scaler == "maxabs":
        X_scaled = MaxAbsScaler().fit_transform(X)
    elif scaler == "standard":
        X_scaled = StandardScaler().fit_transform(X)
    else:
        X_scaled = X

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    labels = pd.Series(cluster_labels, index=customer_item_binary.index, name="Cluster")
    return labels, X_scaled


# ---------- DBSCAN Clustering ----------
def cluster_customers_dbscan(
    customer_item_binary: pd.DataFrame,
    n_pca: int = 20,
    eps: float = 5.0,
    min_samples: int = 5,
    metric: str = "euclidean",
    standardize: bool = True,
    random_state: int = 42,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Input: Customer x Item (boolean) matrix
    Steps: (optional) standardize -> PCA dimensionality reduction -> DBSCAN
    Return: (labels, X_pca)
        labels: pd.Series(index=CustomerID, name='Cluster', dtype=int)  -1 = noise
        X_pca:  reduced features (can be used for plotting)
    """
    X = customer_item_binary.values.astype(float)

    if standardize:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X_pca)
    labels = pd.Series(db.labels_, index=customer_item_binary.index, name="Cluster")
    return labels, X_pca


# ---------- Build Basket x Item Matrix ----------
def build_basket_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert transactions into Order x Item quantity matrix and its binary version
    Return: (basket_qty, basket_binary)
    """
    basket_qty = (df.groupby(["BillNo", "Itemname"])["Quantity"]
                    .sum().unstack().fillna(0).astype(int))
    basket_bin = basket_qty > 0
    return basket_qty, basket_bin


# ---------- FP-Growth within a specific cluster ----------
def rules_for_cluster(
    df_all: pd.DataFrame,
    labels: pd.Series,
    cluster_id: int,
    min_support: float = 0.005,
    min_confidence: float = 0.3,
) -> pd.DataFrame:
    """
    Filter customers belonging to a cluster -> Build basket x item for that cluster -> FP-Growth -> Association rules
    Return columns: ['antecedents','consequents','support','confidence','lift', ...]
    """
    # Select customers in this cluster
    cust_ids = labels.index[labels == cluster_id]
    df_sub = df_all[df_all["CustomerID"].isin(cust_ids)]

    # Basket x Item
    _, basket_bin = build_basket_matrix(df_sub)

    # Frequent itemsets
    itemsets = fpgrowth(basket_bin, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"])

    # Association rules
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return rules


# ---------- Batch generate rules for all clusters ----------
def rules_for_all_clusters(
    df_all: pd.DataFrame,
    labels: pd.Series,
    min_support: float = 0.005,
    min_confidence: float = 0.3,
    exclude_noise: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Generate association rules for each cluster; by default skip noise cluster (-1)
    Return: {cluster_id: rules_df}
    """
    result = {}
    clusters = sorted(labels.unique())
    for cid in clusters:
        if exclude_noise and cid == -1:
            continue
        rules = rules_for_cluster(df_all, labels, cid, min_support, min_confidence)
        result[cid] = rules
    return result
