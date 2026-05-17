from pathlib import Path
from dotenv import load_dotenv
import os
import zipfile
from collections import Counter

load_dotenv()

zip_path_value = os.getenv("STAI_ZIP_FILE")

if not zip_path_value:
    raise ValueError("STAI_ZIP_FILE is missing. Check your .env file.")

zip_path = Path(zip_path_value)

if not zip_path.exists():
    raise FileNotFoundError(f"Zip file not found: {zip_path}")

output_path = Path("reports/zip_file_list.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    files = z.namelist()

    print(f"Zip file: {zip_path}")
    print(f"Total files: {len(files)}")

    print("\nTop-level folders/files:")
    top_levels = Counter(file.split("/")[0] for file in files)
    for name, count in top_levels.most_common():
        print(f"{name}: {count}")

    print("\nFile extensions:")
    extensions = Counter(Path(file).suffix.lower() or "[no extension]" for file in files)
    for ext, count in extensions.most_common():
        print(f"{ext}: {count}")

    with output_path.open("w", encoding="utf-8") as f:
        for file in files:
            f.write(file + "\n")

print(f"\nFull file list saved to: {output_path}")