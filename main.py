import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import cv2

# **Load Tomograms**
def load_all_tomograms(base_path):
    tomograms = {}
    experiment_root = os.path.join(base_path, "train/static/ExperimentRuns")
    
    for experiment in os.listdir(experiment_root):
        zarr_path = os.path.join(experiment_root, experiment, "VoxelSpacing10.000", "denoised.zarr")
        if os.path.exists(zarr_path):
            zarr_store = zarr.open(zarr_path, mode='r')
            if '0' in zarr_store:
                tomogram = np.array(zarr_store['0'])
                tomograms[experiment] = tomogram
                print(f"[INFO] Loaded {experiment}: shape {tomogram.shape}")
            else:
                print(f"[WARNING] Skipping {experiment}: No level 0 found in {zarr_path}")
    return tomograms

# **Load Ground Truth Labels**
def load_labels(base_path, experiment):
    labels_path = os.path.join(base_path, f"train/overlay/ExperimentRuns/{experiment}/Picks")
    if not os.path.exists(labels_path):
        print(f"[WARNING] No labels found for {experiment}")
        return {}

    labels = {}
    for json_file in os.listdir(labels_path):
        if json_file.endswith('.json'):
            particle_type = json_file.replace('.json', '')
            json_file_path = os.path.join(labels_path, json_file)
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            if 'points' in data and isinstance(data['points'], list):
                coords = np.array([[p['location']['x'], p['location']['y'], p['location']['z']] for p in data['points']])
                labels[particle_type] = coords if coords.size > 0 else np.array([])
            else:
                labels[particle_type] = np.array([])
    
    print(f"[DEBUG] {experiment} - Loaded {len(labels)} particle types.")
    return labels

# **Find Valid Slices**
def find_particle_slices(labels_dict, experiment, max_slices):
    particle_slices = {}
    all_slices = set()

    for particle_type, coords in labels_dict[experiment].items():
        if coords.size > 0 and coords.shape[1] == 3:
            valid_slices = np.unique(coords[:, 2]).astype(int)
            valid_slices = valid_slices[valid_slices < max_slices]  # Ensure within bounds
            particle_slices[particle_type] = valid_slices
            all_slices.update(valid_slices)
            print(f"[DEBUG] {experiment} - {particle_type} appears in slices: {valid_slices[:10]} ...")

    return sorted(all_slices)

# **Preprocessing Function**
def preprocess_image(img):
    """Applies histogram equalization and Gaussian blur to enhance image quality."""
    img = (img / np.max(img) * 255).astype(np.uint8)  # Normalize to 0-255
    img_eq = cv2.equalizeHist(img)  # Contrast enhancement
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)  # Noise reduction
    return img_blur

# **Save Slices with Particles (With Preprocessing)**
def save_valid_slices(tomograms, labels_dict, save_dir="output_images"):
    os.makedirs(save_dir, exist_ok=True)

    for experiment in tomograms.keys():
        max_slices = tomograms[experiment].shape[0]  # Get valid slice range
        valid_slices = find_particle_slices(labels_dict, experiment, max_slices)

        if not valid_slices:
            print(f"[INFO] {experiment} - No particles detected.")
            continue

        print(f"[INFO] {experiment} - Saving {len(valid_slices)} preprocessed slices.")

        img_height, img_width = tomograms[experiment].shape[1:3]

        for slice_index in valid_slices:
            img = tomograms[experiment][slice_index, :, :]
            img = preprocess_image(img)  # Apply preprocessing
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f"{experiment} - Slice {slice_index}")
            plt.axis("off")

            # Overlay particle locations
            for particle_type, coords in labels_dict[experiment].items():
                if coords.size > 0 and coords.shape[1] == 3:
                    mask = (coords[:, 2] == slice_index)

                    if np.any(mask):
                        scaled_x = (coords[mask, 0] / np.max(coords[:, 0])) * img_width
                        scaled_y = (coords[mask, 1] / np.max(coords[:, 1])) * img_height
                        plt.scatter(scaled_x, scaled_y, label=particle_type, alpha=0.7, s=100, edgecolors='white')

            plt.legend(fontsize=8, loc='upper right')
            save_path = os.path.join(save_dir, f"{experiment}_slice_{slice_index}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()  # Free memory
            print(f"[INFO] Saved: {save_path}")

# **Run All Steps**
base_dataset_path = sys.argv[1]  # Update with actual dataset path
tomograms = load_all_tomograms(base_dataset_path)
labels_dict = {exp: load_labels(base_dataset_path, exp) for exp in tomograms.keys()}

save_valid_slices(tomograms, labels_dict)
