import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Define paths
data_dir = "../data/test_data/nii_only"
output_dir = "../data/test_data_normalized"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def normalize_image(image):
    """
    Normalize a NIfTI image by scaling voxel intensities to [0, 1].
    """
    image_data = image.get_fdata()  # Get voxel data as a numpy array
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val - min_val == 0:  # Avoid division by zero
        return image
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return nib.Nifti1Image(normalized_data, image.affine, image.header)

# Process each .nii file in the data directory
for file_name in tqdm(os.listdir(data_dir)):
    if file_name.endswith(".nii"):  # Only process .nii files
        file_path = os.path.join(data_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # Load image
        image = nib.load(file_path)

        # Normalize image
        normalized_image = normalize_image(image)

        # Save normalized image
        nib.save(normalized_image, output_path)

print("Preprocessing complete. Normalized files saved in:", output_dir)
