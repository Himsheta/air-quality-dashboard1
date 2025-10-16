import pandas as pd
from pathlib import Path

# Define paths
RAW_FILE = Path("data/raw/city_day.csv")
PROCESSED_FILE = Path("data/processed/clean_data.csv")

# Read the dataset
df = pd.read_csv(RAW_FILE)

# Basic preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['AQI'])
df = df.fillna(0)

# Create processed folder if it doesn't exist
PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save clean data
df.to_csv(PROCESSED_FILE, index=False)
print(f"✅ Clean data saved to {PROCESSED_FILE}")
