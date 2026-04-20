import os
from dotenv import load_dotenv
import cv2
import glob
import shutil
from tqdm import tqdm
from roboflow import Roboflow
from sahi.slicing import slice_image

# --- 1. CONFIGURATION (Constants) ---
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY") # API_KEY = "YOUR_API_KEY"
WORKSPACE = "bahs-work-space" #WORKSPACE = "YOUR_WORKSPACE_NAME"
PROJECT = "metal-defect-detection-neu-gc10-bristol-university" #PROJECT = 1YOUR_PROJECT_NAME"
VERSION = 2

# Path Definitions
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Slicing Parameters
SLICE_SIZE = 640
OVERLAP_RATIO = 0.2


# --- 2. CORE FUNCTIONS ---

def download_from_roboflow():
    """Checks for local data, if not found, downloads from Roboflow."""
    if not os.path.exists(RAW_DATA_DIR):
        print("🚀 Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        version = project.version(VERSION)
        dataset = version.download("yolo26", location=RAW_DATA_DIR)
        return dataset.location
    print("✅ Raw dataset found locally, skipping download.")
    return RAW_DATA_DIR


def process_labels(image_path, label_path, slice_results, output_dir, base_name):
    """
    Transforms global YOLO coordinates to local slice coordinates.
    Safely handles extra columns in label files.
    """
    img = cv2.imread(image_path)
    if img is None: return
    ih, iw, _ = img.shape

    with open(label_path, 'r') as f:
        original_lines = f.readlines()

    for slice_info in slice_results.starting_pixels:
        tx, ty = slice_info  # Top-left (x, y) of the slice
        slice_label_path = os.path.join(output_dir, f"{base_name}_slice_{tx}_{ty}.txt")
        new_annotations = []

        for line in original_lines:
            parts = line.split()
            if len(parts) < 5: continue  # Skip malformed lines

            # DEFENSIVE FIX: Take only the first 5 columns (class, x, y, w, h)
            cls, x_c, y_c, w, h = map(float, parts[:5])

            # Convert to absolute pixels
            abs_x, abs_y = x_c * iw, y_c * ih
            abs_w, abs_h = w * iw, h * ih

            # Check if the center is within the current slice
            if tx <= abs_x <= tx + SLICE_SIZE and ty <= abs_y <= ty + SLICE_SIZE:
                # Calculate local normalized coordinates relative to slice
                new_x = (abs_x - tx) / SLICE_SIZE
                new_y = (abs_y - ty) / SLICE_SIZE
                new_w = abs_w / SLICE_SIZE
                new_h = abs_h / SLICE_SIZE

                # Boundary clipping (ensure [0.0 - 1.0])
                new_x = max(min(new_x, 1.0), 0.0)
                new_y = max(min(new_y, 1.0), 0.0)
                new_w = min(new_w, 1.0)
                new_h = min(new_h, 1.0)

                new_annotations.append(f"{int(cls)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

        # Save label only if the slice contains a defect
        if new_annotations:
            with open(slice_label_path, 'w') as f_out:
                f_out.write("\n".join(new_annotations))


# --- 3. MAIN PIPELINE EXECUTION ---

def run_pipeline():
    raw_path = download_from_roboflow()

    for split in ["train", "valid", "test"]:
        print(f"\n📂 Processing {split.upper()} set...")
        img_dir = os.path.join(raw_path, split, "images")
        out_dir = os.path.join(PROCESSED_DATA_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        image_list = glob.glob(os.path.join(img_dir, "*.jpg"))

        for img_path in tqdm(image_list, desc=f"Progress ({split})"):
            name = os.path.basename(img_path).replace(".jpg", "")
            lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

            if not os.path.exists(lbl_path): continue

            # READ IMAGE TO CHECK DIMENSIONS
            img_temp = cv2.imread(img_path)
            if img_temp is None: continue
            h, w, _ = img_temp.shape

            # EFFICIENCY CHECK: If image is already small or equal to slice size
            if h <= SLICE_SIZE and w <= SLICE_SIZE:
                # Direct Copy to Processed Folder (No slicing needed)
                cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), img_temp)
                shutil.copy(lbl_path, os.path.join(out_dir, f"{name}.txt"))
                continue

            # SLICING: For large images (e.g., Bristol or High-Res sets)
            result = slice_image(
                image=img_path, output_file_name=name, output_dir=out_dir,
                slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
                overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO
            )

            # COORDINATE TRANSFORMATION
            process_labels(img_path, lbl_path, result, out_dir, name)


if __name__ == "__main__":
    run_pipeline()
    print("\n🚀 SUCCESS: Professional dataset is ready in 'data/processed'!")
