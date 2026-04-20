import os
import cv2
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Define the paths for validation and test sets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
FOLDERS_TO_PROCESS = ["valid", "test"]


def to_3_channel_grayscale(image):
    """
    Converts a BGR image to a 3-channel grayscale image.
    Maintains (H, W, 3) shape for YOLO compatibility.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def run_grayscale_standardization():
    for folder in FOLDERS_TO_PROCESS:
        folder_path = os.path.join(DATA_PATH, folder)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}, skipping...")
            continue

        # Search for all image formats
        image_list = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
            image_list.extend(glob.glob(os.path.join(folder_path, ext)))

        print(f"📂 Processing '{folder}' set: {len(image_list)} images found.")

        for img_path in tqdm(image_list, desc=f"Converting {folder}"):
            # Load the original image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert to grayscale (3-channel)
            gray_img = to_3_channel_grayscale(img)

            # Overwrite the original image with the standardized grayscale version
            cv2.imwrite(img_path, gray_img)

    print("\n✨ Grayscale standardization for Valid and Test sets is complete!")


if __name__ == "__main__":
    run_grayscale_standardization()