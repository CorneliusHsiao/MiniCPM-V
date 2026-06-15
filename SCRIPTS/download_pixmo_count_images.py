from pathlib import Path
from hashlib import sha256
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = Path("/wekafs/ict/hanyuanx/Datasets/pixmo-count")
IMAGE_DIR = OUTPUT_DIR / "images"
LOCAL_DATASET_DIR = OUTPUT_DIR / "dataset_with_local_paths"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("allenai/pixmo-count")

# Use SHA256 as the file name because the same image URL may appear
# multiple times in the annotations.
unique_images = {}
for split_name, split in dataset.items():
    for row in split:
        unique_images[row["image_sha256"]] = row["image_url"]

print(f"Total rows: {sum(len(split) for split in dataset.values())}")
print(f"Unique images to download: {len(unique_images)}")

session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

failed = []

for expected_hash, url in tqdm(unique_images.items(), desc="Downloading images"):
    image_path = IMAGE_DIR / f"{expected_hash}.jpg"

    if image_path.exists():
        existing_hash = sha256(image_path.read_bytes()).hexdigest()
        if existing_hash == expected_hash:
            continue
        image_path.unlink()

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        image_bytes = response.content

        actual_hash = sha256(image_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
            )

        image_path.write_bytes(image_bytes)

    except Exception as exc:
        failed.append((expected_hash, url, str(exc)))

# Add local paths into the Hugging Face dataset.
def add_local_path(example):
    example["local_image_path"] = str(
        IMAGE_DIR / f'{example["image_sha256"]}.jpg'
    )
    return example

local_dataset = dataset.map(add_local_path)
local_dataset.save_to_disk(str(LOCAL_DATASET_DIR))

if failed:
    failed_path = OUTPUT_DIR / "failed_downloads.txt"
    with failed_path.open("w") as f:
        for image_hash, url, error in failed:
            f.write(f"{image_hash}\t{url}\t{error}\n")
    print(f"Finished with {len(failed)} failed downloads.")
    print(f"Failure list saved to: {failed_path}")
else:
    print("All images downloaded and verified successfully.")

print(f"Images saved to: {IMAGE_DIR}")
print(f"Dataset with local_image_path saved to: {LOCAL_DATASET_DIR}")
