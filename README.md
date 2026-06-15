# Evolution of Seismic Hotspots in Nepal (1990–2026): A Spatiotemporal Clustering Approach

## Project Overview
This project builds a comprehensive data analysis and machine learning pipeline to investigate the evolution of earthquake activity in Nepal from 1990 to 2026. The objective is to identify seismic hotspots using clustering algorithms, analyze temporal trends, and generate interpretable risk insights for different regions. 

The goal is not to predict future earthquakes, but to understand historical seismic patterns and identify persistent and emerging high-risk zones.

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd MINspire-Earthquake-Project
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the Dataset:**
   - Download the dataset "Nepal Earthquake Seismicity Dataset (1990–2026)" from Kaggle (ID: `amansinghnp/nepal-earthquake-seismicity-dataset-1990-2026`).
   - Place the CSV file inside the `data/` directory (e.g., `data/nepal_earthquake_data.csv`).

5. **Run the Project:**
   ```bash
   python main.py
   ```

## Methodology
The project is divided into the following phases:
1. **Dataset Inspection**: Automatically identifying missing values, data types, and initial data shape.
2. **Data Cleaning**: Handling missing/duplicate records, date parsing, and ensuring correct formats.
3. **Feature Engineering**: Creating temporal features (decade, season) and categorical features (magnitude and depth categories).
4. **Exploratory Data Analysis (EDA)**: Visualizing magnitude distributions, depth relations, and temporal trends.
5. **Spatial Analysis**: Mapping earthquake epicenters geographically.
6. **Seismic Hotspot Detection**: Applying DBSCAN clustering on spatial features to find hotspots, compared alongside K-Means.
7. **Hotspot Evolution Over Time**: Dividing the timeline into distinct decades to observe changes in clustering behavior and hotspot persistence.
8. **Statistical Analysis**: Computing mean metrics over time and checking correlation among variables.
9. **Risk Zone Identification**: Deriving Low, Moderate, and High-risk zones based on cluster density and earthquake features.

## Results Summary
- **Decadal Trends**: 
  - **1990s**: 280 events, Avg Mag: 4.40, Avg Depth: 34.07 km
  - **2000s**: 443 events, Avg Mag: 4.19, Avg Depth: 25.47 km
  - **2010s**: 549 events, Avg Mag: 4.46, Avg Depth: 16.17 km (significant increase in frequency, dominated by the 2015 Gorkha earthquake sequence)
  - **2020s**: 264 events, Avg Mag: 4.46, Avg Depth: 14.02 km
- **Seismic Hotspot Evolution**:
  - Hotspot detection utilizing DBSCAN identified several key cluster zones.
  - The central-eastern region near Kathmandu/Gorkha is the most active and persistent hotspot across all decades.
- **Risk Zones**:
  - **High Risk**:
    - **Cluster 0**: 1,004 events, Avg Mag: 4.44, Max Mag: 7.8 (Gorkha region)
    - **Cluster 1**: 212 events, Avg Mag: 4.15, Max Mag: 6.7
  - **Low Risk**:
    - **Cluster 2**: 29 events, Max Mag: 5.1
    - **Cluster 3**: 200 events, Max Mag: 5.9
    - **Cluster 4**: 9 events, Max Mag: 4.6
- **Outputs**: All generated plots (such as `dbscan_hotspots.png`, `hotspot_evolution.png`, `correlation_matrix.png`, etc.) and CSV reports are saved automatically inside the `outputs/` folder.

## Future Work
- Integration of predictive modeling (e.g., sequence models for estimating probabilities of aftershocks).
- Real-time API integration to dynamically pull the latest earthquake data and update clusters.
- Overlaying cluster maps with population density data to assess human impact/vulnerability.
