import os
import subprocess

# Dataset identifier
DATASET = "aletbm/aerial-imagery-dataset-floodnet-challenge"

# Output directory
OUT_DIR = os.path.join(os.getcwd(), "floodnet_data")
os.makedirs(OUT_DIR, exist_ok=True)

# How many images you want
N = 1000

# 1. Get list of files using Kaggle CLI
print("Fetching file list from Kaggle...")
result = subprocess.run(
    ["kaggle", "datasets", "files", DATASET],
    capture_output=True, text=True
)

lines = result.stdout.splitlines()

# Skip header rows (first 2 lines are table header/separator)
files = []
for line in lines[2:]:
    parts = line.strip().split()
    if len(parts) > 0 and parts[0].endswith(".jpg"):
        # filename is the first column
        files.append(" ".join(parts[:-2]))  # join name parts before size/date columns

print(f"Found {len(files)} images total")

# 2. Download first N images
for i, fname in enumerate(files[:N]):
    print(f"Downloading {i+1}/{N}: {fname}")
    os.system(f'kaggle datasets download -d {DATASET} -f "{fname}" -p "{OUT_DIR}" --quiet')

print(f"✅ Downloaded {min(N, len(files))} images into {OUT_DIR}")
