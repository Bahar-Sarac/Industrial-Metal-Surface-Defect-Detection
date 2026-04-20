import torch
from ultralytics import YOLO
import os


def main():
    # 1. HARDWARE INITIALIZATION
    # Verify CUDA availability for GPU-accelerated training on RTX 3050
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Hardware Status: NVIDIA GPU Detected ({device_name})")
        selected_device = 0
    else:
        print("⚠️ Warning: GPU not found. Falling back to CPU.")
        selected_device = 'cpu'

    # 2. MODEL DEFINITION
    # Initialize the YOLO26 Nano model.
    # Nano is ideal for the 4GB VRAM limit while maintaining high inference speeds.
    model = YOLO('yolo26s.pt')

    # 3. TRAINING EXECUTION
    # Advanced parameters are tuned for surface defect detection on metal.
    results = model.train(
        # Essential Paths
        data=r'C:/PycharmProjects/metal_defect_detection/data/raw/data.yaml',  # Path to the dataset configuration file
        epochs=150,  # Number of complete passes through the dataset
        imgsz=640,  # Input image size (must match slicing dimensions)
        batch=16,  # Batch size (optimized for 4GB VRAM; reduce to 8 if OOM occurs)
        device=selected_device,  # Target device (GPU index 0)
        workers=4,  # Number of data loading threads (recommended for Windows)

        # Optimization & Convergence
        optimizer='AdamW',  # Adam with Weight Decay for improved stability in fine details
        lr0=0.01,  # Initial learning rate
        label_smoothing=0.1, # Prevents the model from being "overconfident" and making mistakes, increases map ping (mAP).
        patience=20,  # Early stopping: halts training if no improvement after 20 epochs

        # Advanced Data Augmentation (Crucial for Metal Texture Analysis)
        mosaic=1.0,  # Combines 4 images into one; improves small object detection
        mixup=0.5,  # Blends images to enhance model generalization
        degrees=15.0,  # Random rotation to handle varying defect orientations
        fliplr=0.5,  # 50% chance of horizontal flip
        flipud=0.5,  # 50% chance of vertical flip

        # Logging & Artifact Management
        project='metal_defect_detection',  # Root directory for all training runs
        name='yolo26_grayscale_optimized_v2',  # Unique identifier for this experimental run
        exist_ok=True,  # Overwrite existing folders with the same name
        plots=True  # Generate training curves and confusion matrices automatically
    )

    print("\n" + "=" * 60)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print(f"📍 Best weights saved to: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
    print("=" * 60)


if __name__ == '__main__':
    # Required for Windows to prevent infinite subprocess loops
    main()