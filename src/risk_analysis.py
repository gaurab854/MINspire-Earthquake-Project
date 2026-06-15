import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def identify_risk_zones(df, output_dir):
    """Phase 9: Risk Zone Identification"""
    logging.info("Identifying Risk Zones...")
    
    if 'DBSCAN_Cluster' not in df.columns:
        logging.warning("DBSCAN clusters not found. Run clustering first.")
        return
        
    # Analyze clusters (ignoring noise -1)
    cluster_df = df[df['DBSCAN_Cluster'] != -1]
    
    if cluster_df.empty:
        logging.warning("No clusters found. Risk zone identification skipped.")
        return
        
    # Aggregate data by cluster
    risk_summary = cluster_df.groupby('DBSCAN_Cluster').agg(
        Earthquake_Count=('Magnitude', 'count'),
        Average_Magnitude=('Magnitude', 'mean'),
        Max_Magnitude=('Magnitude', 'max'),
        Average_Depth=('Depth', 'mean')
    ).reset_index()
    
    # Define Risk Levels based on logic
    # Logic: 
    # High Risk: Count > 75th percentile OR Max_Magnitude >= 6.0
    # Moderate Risk: Count > median OR Average_Magnitude >= 4.5
    # Low Risk: Others
    
    count_75 = risk_summary['Earthquake_Count'].quantile(0.75)
    count_50 = risk_summary['Earthquake_Count'].median()
    
    def assign_risk(row):
        if row['Earthquake_Count'] > count_75 or row['Max_Magnitude'] >= 6.0:
            return 'High Risk'
        elif row['Earthquake_Count'] > count_50 or row['Average_Magnitude'] >= 4.5:
            return 'Moderate Risk'
        else:
            return 'Low Risk'
            
    risk_summary['Risk_Level'] = risk_summary.apply(assign_risk, axis=1)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'risk_zones_summary.csv')
    risk_summary.to_csv(csv_path, index=False)
    
    # Write summary text
    with open(os.path.join(output_dir, 'risk_zones_report.txt'), 'w', encoding="utf-8") as f:
        f.write("=== Risk Zone Identification ===\n\n")
        f.write("Methodology:\n")
        f.write("- High Risk: High earthquake frequency (top 25% of clusters) OR history of strong earthquakes (Max Mag >= 6.0).\n")
        f.write("- Moderate Risk: Moderate frequency (above median) OR moderate average magnitude (Avg Mag >= 4.5).\n")
        f.write("- Low Risk: Clusters that do not meet the above criteria.\n\n")
        
        f.write("Risk Zone Breakdown:\n")
        f.write(f"{risk_summary.to_string()}\n\n")
        
    logging.info(f"Risk zones identified and saved to {csv_path}")

