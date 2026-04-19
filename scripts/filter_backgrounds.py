import os
import cv2
import glob
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "train")
TARGET_BG_PERCENT = 0.20


def filter_backgrounds():
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ Folder not found: {PROCESSED_DIR}")
        return

    # Let's find all the images (jpg, JPG, png, it doesn't matter)
    all_images = []
    for ext in ["*.jpg", "*.JPG", "*.png", "*.jpeg"]:
        all_images.extend(glob.glob(os.path.join(PROCESSED_DIR, ext)))

    labeled_images = []
    empty_images = []

    print(f"🔍 Checking a total of {len(all_images)} images...")

    for img_path in all_images:
        # Separate the file name and extension
        base = os.path.splitext(img_path)[0]
        lbl_path = base + ".txt"

        if os.path.exists(lbl_path):
            labeled_images.append(img_path)
        else:
            empty_images.append(img_path)

    num_labeled = len(labeled_images)
    num_empty = len(empty_images)

    print(f"📊 Current Status:\n - Labeled (Incorrect): {num_labeled}\n - Empty (Clean): {num_empty}")

    if num_empty == 0:
        print("\n⚠️ WARNING: No empty images were found. Could the slicing process not be complete yet?")
        return

    # Target blank image count calculation
    total_target = int(num_labeled / (1 - TARGET_BG_PERCENT))
    allowed_empty_count = total_target - num_labeled

    if allowed_empty_count >= num_empty:
        print("✅ The number of blank images is already quite low.")
        return

    # Selecting the best blank images with the Variance Filter
    print(f"🧠 The best {allowed_empty_count} empty images are selected from a {num_empty} set of empty images...")
    empty_with_variance = []
    for img_path in tqdm(empty_images):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        empty_with_variance.append((img_path, variance))

    empty_with_variance.sort(key=lambda x: x[1], reverse=True)
    delete_empty = [x[0] for x in empty_with_variance[allowed_empty_count:]]

    print(f"🗑️ {len(delete_empty)} unnecessary empty images are being deleted...")
    for img_path in tqdm(delete_empty):
        os.remove(img_path)

    print(f"✨ Filtering complete! Remaining total: {num_labeled + allowed_empty_count}")


if __name__ == "__main__":
    filter_backgrounds()