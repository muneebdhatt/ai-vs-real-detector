"""Download CIFAKE from Kaggle and print the folder to use as DATA_DIR.

Needs a free Kaggle account. Either:
  - put kaggle.json in ~/.kaggle/ , or
  - set KAGGLE_USERNAME and KAGGLE_KEY environment variables.
In Colab: upload kaggle.json first, or use the Kaggle notebook where the dataset can be attached directly.

Usage:
    python src/download_data.py
Then:
    export DATA_DIR=<printed path>     (Windows PowerShell: $env:DATA_DIR="<printed path>")
"""
import os
import kagglehub

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Downloaded to:", path)

# Expected structure: <path>/train/{REAL,FAKE} and <path>/test/{REAL,FAKE}
for split in ("train", "test"):
    for cls in ("REAL", "FAKE"):
        d = os.path.join(path, split, cls)
        n = len(os.listdir(d)) if os.path.isdir(d) else "MISSING"
        print(f"{split}/{cls}: {n}")
print("\nSet DATA_DIR to:", path)
