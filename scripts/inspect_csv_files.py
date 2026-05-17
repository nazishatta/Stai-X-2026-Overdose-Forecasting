from pathlib import Path
from dotenv import load_dotenv
import os
import zipfile
import pandas as pd

load_dotenv()

zip_path_value = os.getenv("STAI_ZIP_FILE")

if not zip_path_value:
    raise ValueError("STAI_ZIP_FILE is missing. Check your .env file.")

zip_path = Path(zip_path_value)

if not zip_path.exists():
    raise FileNotFoundError(f"Zip file not found: {zip_path}")

csv_files = [
    "sample_submission.csv",
    "train/covariates.csv",
    "train/dose_sys_train.csv",
    "val/covariates.csv",
]

with zipfile.ZipFile(zip_path, "r") as z:
    for csv_file in csv_files:
        print("\n" + "=" * 80)
        print(f"FILE: {csv_file}")
        print("=" * 80)

        with z.open(csv_file) as f:
            df = pd.read_csv(f)

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nMissing values:")
        print(df.isna().sum())

        print("\nDtypes:")
        print(df.dtypes)