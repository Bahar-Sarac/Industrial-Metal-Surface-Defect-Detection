import os
import cv2
import glob
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
import shutil

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "processed", "train")

# Class IDs (Please update this according to your classes.txt order!)
# Example: {'crease_and_fold': 2, 'hole': 4, ...}
CLASS_MAP = {
    2: 8.0,  # crease_and_fold -> 8x
    8: 2.5,  # rolled_in_scale -> 2.5x
    4: 3,  # hole -> 3x
    1: 2.0  # crazing -> 2x
}

# --- 2. AUGMENTATION PIPELINE ---
#This pipeline automatically converts both images and bboxes.
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def to_grayscale(image):
    """Converts image to 3-channel grayscale to maintain YOLO input compatibility."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def process_and_balance():
    all_images = glob.glob(os.path.join(TRAIN_DIR, "*.jpg"))
    print(f"🚀 A total of {len(all_images)} images are being processed...")

    for img_path in tqdm(all_images):
        base_path = os.path.splitext(img_path)[0]
        lbl_path = base_path + ".txt"

        if not os.path.exists(lbl_path): continue

        # 1. Upload the original image and grayscale it (standardize).
        image = cv2.imread(img_path)
        if image is None: continue
        image = to_grayscale(image)
        cv2.imwrite(img_path, image)  # Orijinal resmi gri haliyle üstüne yaz

        # 2. Read the labels.
        bboxes = []
        class_labels = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                cls = int(parts[0])
                bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                class_labels.append(cls)

        # 3. Determine which multiplier to use (If there is more than one class in the image, use the largest one)
        multipliers = [CLASS_MAP.get(cls, 1.0) for cls in class_labels]
        max_multiplier = max(multipliers)

        if max_multiplier > 1.0:
            # How many additional copies will be produced? (e.g., 1 original + 1.5 additional copies for 2.5x)
            # Produce the exact number of copies for the whole part, leave the decimal part to chance.
            num_to_create = int(max_multiplier - 1)
            decimal_part = (max_multiplier - 1) - num_to_create
            if random.random() < decimal_part:
                num_to_create += 1

            for i in range(num_to_create):
                # Apply augmentation
                transformed = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']

                # Save with the new name
                aug_name = f"{os.path.basename(base_path)}_aug_{i}"
                aug_img_path = os.path.join(TRAIN_DIR, f"{aug_name}.jpg")
                aug_lbl_path = os.path.join(TRAIN_DIR, f"{aug_name}.txt")

                cv2.imwrite(aug_img_path, aug_img)
                with open(aug_lbl_path, 'w') as f_out:
                    for j in range(len(aug_bboxes)):
                        b = aug_bboxes[j]
                        f_out.write(f"{class_labels[j]} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

    print("\n✨ Grayscale conversion and balancing are complete!")


if __name__ == "__main__":
    process_and_balance()