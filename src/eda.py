import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def perform_eda(df, output_dir):
    """Phase 4: Exploratory Data Analysis & Phase 5: Spatial Analysis"""
    logging.info("Starting Exploratory Data Analysis (EDA)...")
    
    sns.set_theme(style="whitegrid")
    
    # --- 1. Magnitude Analysis ---
    if 'Magnitude' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Magnitude'], bins=30, kde=True, color='red')
        plt.title('Magnitude Distribution')
        plt.xlabel('Magnitude')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'magnitude_distribution.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Magnitude'], color='orange')
        plt.title('Magnitude Boxplot')
        plt.xlabel('Magnitude')
        plt.savefig(os.path.join(output_dir, 'magnitude_boxplot.png'))
        plt.close()
        
    if 'Mag_Category' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='Mag_Category', order=['Minor', 'Moderate', 'Strong'], palette='viridis')
        plt.title('Earthquakes by Magnitude Category')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'magnitude_categories.png'))
        plt.close()

    # --- 2. Depth Analysis ---
    if 'Depth' in df.columns and 'Magnitude' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Depth'], bins=30, kde=True, color='blue')
        plt.title('Depth Distribution')
        plt.xlabel('Depth (km)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'depth_distribution.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Depth', y='Magnitude', alpha=0.5, color='purple')
        plt.title('Depth vs Magnitude')
        plt.xlabel('Depth (km)')
        plt.ylabel('Magnitude')
        plt.savefig(os.path.join(output_dir, 'depth_vs_magnitude.png'))
        plt.close()

    # --- 3. Temporal Analysis ---
    if 'Year' in df.columns:
        yearly_counts = df['Year'].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='darkblue')
        plt.title('Earthquakes per Year (1990 - 2026)')
        plt.xlabel('Year')
        plt.ylabel('Number of Earthquakes')
        
        # Highlight 2015
        if 2015 in yearly_counts.index:
            plt.axvline(x=2015, color='red', linestyle='--', label='2015 Gorkha Earthquake')
            plt.legend()
            
        plt.savefig(os.path.join(output_dir, 'earthquakes_per_year.png'))
        plt.close()

    if 'Month' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Month', palette='coolwarm')
        plt.title('Earthquakes per Month')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'earthquakes_per_month.png'))
        plt.close()

    # --- 4. Spatial Analysis ---
    logging.info("Generating Spatial Visualizations...")
    if 'Longitude' in df.columns and 'Latitude' in df.columns and 'Magnitude' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df, 
            x='Longitude', 
            y='Latitude', 
            size='Magnitude',
            hue='Magnitude',
            sizes=(10, 200),
            palette='flare',
            alpha=0.6
        )
        plt.title('Geographic Distribution of Epicenters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spatial_distribution.png'))
        plt.close()
        
    logging.info("EDA completed. Visualizations saved.")

