import pandas as pd
import numpy as np
import os

def process_data():

    # load data
    file_path = 'forestfires.csv'
    df = pd.read_csv(file_path)
    
    # Initial stats
    print(f"Original shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # 1. Data Cleaning: Duplicates
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows.")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Duplicates removed. New shape: {df.shape}")

    # 2. Feature Engineering: Encoding Categorical Variables
    # Month mapping
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    # Day mapping
    day_map = {
        'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7
    }
    
    df['month_idx'] = df['month'].map(month_map)
    df['day_idx'] = df['day'].map(day_map)
    
    # Verify mapping
    if df['month_idx'].isnull().any() or df['day_idx'].isnull().any():
        print("Error in mapping months or days.")
    
    # 3. Target Transformation
    # The paper suggests log(area + 1)
    df['area_log'] = np.log1p(df['area'])
    
    # 4. Save cleaned dataset with all columns (for inspection/flexibility)
    output_file_all_cols = 'forestfires_cleaned.csv'
    df.to_csv(output_file_all_cols, index=False)
    print(f"Cleaned dataset with all columns saved to {output_file_all_cols}")
    print(df.head())

if __name__ == "__main__":
    process_data()
