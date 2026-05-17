from pathlib import Path
from dotenv import load_dotenv
import os
import zipfile

load_dotenv()

zip_path = Path(os.getenv("STAI_ZIP_FILE"))

if not zip_path.exists():
    raise FileNotFoundError(f"Zip file not found: {zip_path}")

print("Zip file found:")
print(zip_path)

print("\nFiles inside zip:")
with zipfile.ZipFile(zip_path, "r") as z:
    for file in z.namelist():
        print(file)