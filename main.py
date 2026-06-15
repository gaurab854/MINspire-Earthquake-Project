import os
import pandas as pd
import logging
from src.config import DATA_DIR, OUTPUT_DIR
from src.data_processing import get_data_path, inspect_dataset, clean_data, engineer_features
from src.eda import perform_eda
from src.clustering import detect_hotspots, analyze_hotspot_evolution
from src.statistical_analysis import perform_statistical_analysis
from src.risk_analysis import identify_risk_zones

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    logging.info("Starting MINspire Earthquake Project Pipeline...")
    
    # 1. Get Data Path
    try:
        data_path = get_data_path(DATA_DIR)
        logging.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        logging.error(e)
        return
        
    # Phase 1: Dataset Inspection
    inspect_dataset(df, OUTPUT_DIR)
    
    # Phase 2: Data Cleaning
    df_clean = clean_data(df)
    
    # Phase 3: Feature Engineering
    df_engineered = engineer_features(df_clean)
    
    # Save cleaned data
    clean_data_path = os.path.join(OUTPUT_DIR, "cleaned_earthquake_data.csv")
    df_engineered.to_csv(clean_data_path, index=False)
    logging.info(f"Cleaned and feature-engineered dataset saved to {clean_data_path}")
    
    # Phase 4 & 5: Exploratory Data Analysis & Spatial Analysis
    perform_eda(df_engineered, OUTPUT_DIR)
    
    # Phase 6: Seismic Hotspot Detection
    df_clustered = detect_hotspots(df_engineered, OUTPUT_DIR)
    
    # Phase 7: Hotspot Evolution Over Time
    df_evolved = analyze_hotspot_evolution(df_clustered, OUTPUT_DIR)
    
    # Phase 8: Statistical Analysis
    perform_statistical_analysis(df_evolved, OUTPUT_DIR)
    
    # Phase 9: Risk Zone Identification
    identify_risk_zones(df_evolved, OUTPUT_DIR)
    
    logging.info("Pipeline completed successfully! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()
