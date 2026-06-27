import urllib.request
import os

def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    raw_dir = r"f:\GITHUB\h_ds\heart-disease-ai\data\raw"
    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, "cleveland.csv")
    
    print(f"Downloading from {url} to {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    print("Download completed.")

if __name__ == "__main__":
    download_dataset()
