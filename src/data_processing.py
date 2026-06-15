import os
import io
import shutil
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load .env from project root (gitignored) - works regardless of where this file is run from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
except ImportError:
    pass  # dotenv not installed; user must set KAGGLE_DATASET_ID manually

def get_data_path(data_dir):
    """Finds the first CSV file in the data directory. If not found, downloads it via kagglehub."""
    # Check if a CSV file already exists in data_dir
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            return os.path.join(data_dir, csv_files[0])
    
    # If not found, download via kagglehub
    logging.info("Local dataset not found. Downloading via kagglehub...")
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is not installed. Run: pip install kagglehub")
    
    try:
        dataset_id = os.environ.get("KAGGLE_DATASET_ID")
        if not dataset_id:
            raise ValueError("KAGGLE_DATASET_ID not set. Add it to your .env file (see .env.example).")
        download_path = kagglehub.dataset_download(dataset_id)
        logging.info(f"Dataset downloaded to cache path: {download_path}")
        
        # Find all CSV files in the downloaded path
        downloaded_files = os.listdir(download_path)
        csv_files = [f for f in downloaded_files if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the downloaded kagglehub dataset path: {download_path}")
        
        # Copy the first CSV to our data directory
        os.makedirs(data_dir, exist_ok=True)
        dest_path = os.path.join(data_dir, csv_files[0])
        src_path = os.path.join(download_path, csv_files[0])
        shutil.copy(src_path, dest_path)
        logging.info(f"Successfully copied dataset to local path: {dest_path}")
        return dest_path
    except Exception as e:
        logging.error(f"Failed to automatically download dataset: {e}")
        raise FileNotFoundError(f"Could not load or download the dataset. Please ensure you have an active internet connection or manually place the CSV in {data_dir}.")

def inspect_dataset(df, output_dir):
    """Phase 1: Dataset Inspection"""
    logging.info("Starting Dataset Inspection...")
    
    with open(os.path.join(output_dir, "dataset_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Dataset Inspection Report ===\n\n")
        
        f.write(f"1. Dataset Shape: {df.shape}\n\n")
        
        f.write("2. Head (First 5 Rows):\n")
        f.write(f"{df.head().to_string()}\n\n")
        
        f.write("3. Info and Data Types:\n")
        # capture df.info()
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(f"{buffer.getvalue()}\n\n")
        
        f.write("4. Summary Statistics (Describe):\n")
        f.write(f"{df.describe(include='all').to_string()}\n\n")
        
        f.write("5. Missing Values:\n")
        f.write(f"{df.isnull().sum().to_string()}\n\n")
        
        f.write("6. Duplicate Rows:\n")
        f.write(f"{df.duplicated().sum()}\n")
        
    logging.info(f"Dataset inspection report saved to {output_dir}/dataset_summary.txt")


def clean_data(df):
    """Phase 2: Data Cleaning"""
    logging.info("Starting Data Cleaning...")
    
    # 1. Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    logging.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    # Identify possible column names dynamically based on typical Nepal earthquake datasets
    # Usually: time/Date, latitude/Lat, longitude/Long, depth/Depth, mag/Magnitude
    
    col_mapping = {}
    # 1. Exact or common prefix/suffix matching
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in ['time', 'date', 'datetime']:
            col_mapping['Datetime'] = col
        elif c_lower in ['latitude', 'lat']:
            col_mapping['Latitude'] = col
        elif c_lower in ['longitude', 'lon', 'lng']:
            col_mapping['Longitude'] = col
        elif c_lower == 'depth':
            col_mapping['Depth'] = col
        elif c_lower in ['mag', 'magnitude']:
            col_mapping['Magnitude'] = col
            
    # 2. Fallback to substring matching for missing columns
    for col in df.columns:
        c_lower = col.lower()
        if 'Datetime' not in col_mapping and ('time' in c_lower or 'date' in c_lower):
            col_mapping['Datetime'] = col
        elif 'Latitude' not in col_mapping and 'lat' in c_lower:
            col_mapping['Latitude'] = col
        elif 'Longitude' not in col_mapping and ('lon' in c_lower or 'lng' in c_lower):
            col_mapping['Longitude'] = col
        elif 'Depth' not in col_mapping and 'depth' in c_lower:
            col_mapping['Depth'] = col
        elif 'Magnitude' not in col_mapping and ('mag' in c_lower and 'type' not in c_lower and 'nst' not in c_lower and 'source' not in c_lower and 'error' not in c_lower):
            col_mapping['Magnitude'] = col
            
    # Standardize column names
    rename_dict = {v: k for k, v in col_mapping.items()}
    df = df.rename(columns=rename_dict)
    
    # Drop rows missing crucial info
    cols_to_check = ['Datetime', 'Latitude', 'Longitude', 'Magnitude']
    existing_cols_to_check = [c for c in cols_to_check if c in df.columns]
    df = df.dropna(subset=existing_cols_to_check)
    
    # Convert types
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime']) # Drop if parsing failed
        
    for numeric_col in ['Latitude', 'Longitude', 'Depth', 'Magnitude']:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
            
    df = df.dropna(subset=[c for c in ['Latitude', 'Longitude', 'Magnitude'] if c in df.columns])
    
    # Handle missing depth (fill with median)
    if 'Depth' in df.columns:
        df['Depth'] = df['Depth'].fillna(df['Depth'].median())
    
    logging.info("Data cleaning completed.")
    return df


def _get_season(month):
    # Winter: Dec, Jan, Feb
    # Pre-monsoon: Mar, Apr, May
    # Monsoon: Jun, Jul, Aug, Sep
    # Post-monsoon: Oct, Nov
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Pre-monsoon'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Post-monsoon'


def _get_mag_category(mag):
    if mag < 4.0: return 'Minor'
    elif mag <= 5.0: return 'Moderate'
    else: return 'Strong'


def _get_depth_category(depth):
    # Common thresholds: < 70 Shallow, 70-300 Intermediate, >300 Deep
    if depth < 70: return 'Shallow'
    elif depth <= 300: return 'Intermediate'
    else: return 'Deep'


def engineer_features(df):
    """Phase 3: Feature Engineering"""
    logging.info("Starting Feature Engineering...")
    
    if 'Datetime' in df.columns:
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        df['Day'] = df['Datetime'].dt.day
        df['Hour'] = df['Datetime'].dt.hour
        df['Decade'] = (df['Year'] // 10) * 10
        df['Season'] = df['Month'].apply(_get_season)
        
    if 'Magnitude' in df.columns:
        df['Mag_Category'] = df['Magnitude'].apply(_get_mag_category)
        
    if 'Depth' in df.columns:
        df['Depth_Category'] = df['Depth'].apply(_get_depth_category)
        
    logging.info("Feature engineering completed.")
    return df

