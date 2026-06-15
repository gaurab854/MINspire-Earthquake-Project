import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def detect_hotspots(df, output_dir):
    """Phase 6: Seismic Hotspot Detection"""
    logging.info("Starting Seismic Hotspot Detection...")
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        logging.warning("Latitude or Longitude missing. Cannot perform spatial clustering.")
        return df

    coords = df[['Latitude', 'Longitude']].values
    
    # Standardize for distance metrics
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # --- DBSCAN Clustering ---
    # Epsilon and min_samples might need tuning based on actual data
    dbscan = DBSCAN(eps=0.2, min_samples=10)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(coords_scaled)
    
    # Visualize DBSCAN
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='Longitude', y='Latitude', 
        hue='DBSCAN_Cluster', palette='tab20', alpha=0.6,
        legend='full'
    )
    plt.title('DBSCAN Seismic Hotspots (Noise is -1)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'dbscan_hotspots.png'))
    plt.close()

    # --- KMeans Clustering (For comparison) ---
    # Use Elbow method to find optimal K
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(coords_scaled)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o')
    plt.title('K-Means Elbow Method')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(output_dir, 'kmeans_elbow.png'))
    plt.close()
    
    # Fit KMeans with a typical optimal k, say K=4
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['KMeans_Cluster'] = kmeans.fit_predict(coords_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='Longitude', y='Latitude', 
        hue='KMeans_Cluster', palette='Set1', alpha=0.6,
        legend='full'
    )
    plt.title(f'K-Means Clustering (K={optimal_k})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'kmeans_hotspots.png'))
    plt.close()
    
    # Explanation written to text file
    with open(os.path.join(output_dir, "clustering_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("=== DBSCAN vs K-Means for Seismic Hotspots ===\n\n")
        f.write("DBSCAN is generally better suited for earthquake epicenter clustering because:\n")
        f.write("1. It does not require specifying the number of clusters in advance.\n")
        f.write("2. It can find arbitrarily shaped clusters (fault lines are often linear/curved, not spherical like K-Means assumes).\n")
        f.write("3. It has a notion of 'noise' (cluster -1), which handles scattered, independent seismic events naturally.\n")

    return df


def analyze_hotspot_evolution(df, output_dir):
    """Phase 7: Hotspot Evolution Over Time"""
    logging.info("Analyzing Hotspot Evolution Over Time...")
    
    if 'Decade' not in df.columns:
        return df
    
    # We will look at distinct periods: 1990-1999, 2000-2009, 2010-2019, 2020-2026
    # Create periods
    def get_period(year):
        if 1990 <= year <= 1999: return '1990-1999'
        elif 2000 <= year <= 2009: return '2000-2009'
        elif 2010 <= year <= 2019: return '2010-2019'
        elif year >= 2020: return '2020-2026'
        else: return 'Other'
        
    df['Period'] = df['Year'].apply(get_period)
    periods = ['1990-1999', '2000-2009', '2010-2019', '2020-2026']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, period in enumerate(periods):
        period_df = df[df['Period'] == period].copy()
        ax = axes[idx]
        
        if len(period_df) < 10:
            ax.set_title(f'{period} (Insufficient data)')
            continue
            
        coords = period_df[['Latitude', 'Longitude']].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        dbscan = DBSCAN(eps=0.25, min_samples=5) # Adjusted for smaller subsets
        clusters = dbscan.fit_predict(coords_scaled)
        period_df['Period_Cluster'] = clusters
        
        # Count non-noise clusters
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        sns.scatterplot(
            data=period_df, x='Longitude', y='Latitude', 
            hue='Period_Cluster', palette='tab10', alpha=0.7, ax=ax, legend=False
        )
        ax.set_title(f'{period} | Clusters: {n_clusters} | Events: {len(period_df)}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hotspot_evolution.png'))
    plt.close()
    
    logging.info("Hotspot evolution analysis completed.")
    return df
