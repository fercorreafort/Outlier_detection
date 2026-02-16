import kagglehub
import pandas as pd
import os

# Download the dataset
path = kagglehub.dataset_download("victorsoeiro/hbo-max-tv-shows-and-movies")
print(f"Path to dataset files: {path}")

# List all files in the downloaded directory
print("\nFiles in dataset:")
files = os.listdir(path)
for file in files:
    print(f"  - {file}")

# Find and load the CSV file
csv_files = [f for f in files if f.endswith('.csv')]
if csv_files:
    csv_file = os.path.join(path, csv_files[0])
    HBO = pd.read_csv(csv_file)
    print(f"\nLoaded: {csv_files[0]}")
    
    print("\nFirst 5 records:")
    print(HBO.head())
    
    print("\nDataset Info:")
    print(HBO.info())
    
    print("\nDataset Shape:")
    print(f"Rows: {HBO.shape[0]}, Columns: {HBO.shape[1]}")
    
    print("\nColumn Names:")
    print(HBO.columns.tolist())
    
    print("\nBasic Statistics:")
    print(HBO.describe(include='all'))
    
    print("\nMissing Values:")
    print(HBO.isnull().sum())
else:
    print("\nNo CSV files found. Available files:")
    print(files)