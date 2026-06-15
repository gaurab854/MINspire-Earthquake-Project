import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def perform_statistical_analysis(df, output_dir):
    """Phase 8: Statistical Analysis"""
    logging.info("Starting Statistical Analysis...")
    
    with open(os.path.join(output_dir, "statistical_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Statistical Analysis Summary ===\n\n")
        
        # Mean metrics by decade
        if 'Decade' in df.columns:
            decade_stats = df.groupby('Decade').agg({
                'Magnitude': ['mean', 'max', 'count'],
                'Depth': 'mean'
            }).round(2)
            
            f.write("1. Statistics by Decade:\n")
            f.write(f"{decade_stats.to_string()}\n\n")
            
        # Annual earthquake frequency
        if 'Year' in df.columns:
            annual_freq = df['Year'].value_counts().sort_index()
            f.write(f"2. Average Earthquakes per Year: {annual_freq.mean():.2f}\n\n")
            
            # Correlation Matrix
            f.write("3. Correlation Matrix:\n")
            numeric_cols = ['Magnitude', 'Depth', 'Latitude', 'Longitude']
            # Add temporal features if numeric
            if 'Year' in df.columns: numeric_cols.append('Year')
            
            existing_cols = [c for c in numeric_cols if c in df.columns]
            corr_matrix = df[existing_cols].corr()
            f.write(f"{corr_matrix.to_string()}\n\n")
            
            # Visualize correlation matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
            plt.close()
            
    logging.info("Statistical analysis completed.")
