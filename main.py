import os
from roboflow import Roboflow
from sahi.slicing import slice_image
from glob import glob
from tqdm import tqdm

# --- 1. SETTINGS ---
API_KEY = "YOUR-API-KEY"
WORKSPACE = "workspace-name"
PROJECT = "project-name"
VERSION = 1

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# --- 2. DOWNLOAD FROM ROBOFLOW ---
def download_from_roboflow():
    if not os.path.exists(RAW_DATA_DIR):
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        version = project.version(VERSION)
        dataset = version.download("yolo26", location=RAW_DATA_DIR)
        return dataset.location
    return RAW_DATA_DIR


# --- 3. CLICING WITH SAHI ---
def slice_dataset(input_dir, output_dir):
    # Get visual paths
    image_list = glob(os.path.join(input_dir, "*.jpg"))

    print(f"\nThe slicing process begins: {len(image_list)} images were found.")

    for image_path in tqdm(image_list):
        # Find the path to the label file (move the images -> labels folder)
        # In the YOLO structure, the images and labels folders are at the same level.
        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")

        # SAHI may give an error if the tag file is missing; it should be checked.
        if not os.path.exists(label_path):
            continue

        slice_image(
            image=image_path,
            output_file_name=os.path.basename(image_path).replace(".jpg", ""),
            output_dir=output_dir,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            min_area_ratio=0.3,
            # CRITICAL ADDITION: To also process tags in YOLO format.
            annotation_original_path=label_path,
            export_format="yolo"
        )


if __name__ == "__main__":
    raw_path = download_from_roboflow()

    for split in ["train", "valid", "test"]:
        print(f"\n--- {split.upper()} THE SET IS BEING PROCESSED ---")
        input_split_dir = os.path.join(raw_path, split, "images")
        output_split_dir = os.path.join(PROCESSED_DATA_DIR, split)

        os.makedirs(output_split_dir, exist_ok=True)
        slice_dataset(input_split_dir, output_split_dir)

print("\n🚀 All transactions were completed successfully! The dataset is ready in the 'data/processed' folder.")